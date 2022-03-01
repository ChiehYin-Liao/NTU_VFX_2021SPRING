#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd
from scipy.spatial.distance import cdist


# In[2]:


def read_img(dirname):
    rgbs = []
    for filename in np.sort(os.listdir(dirname)):
        if osp.splitext(filename)[1] in ['.jpg', '.png', '.JPG']:
            bgr = cv2.imread(osp.join(dirname,filename))
            rgbs += [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)]
            
    return np.array(rgbs) 


# In[3]:


def read_focal(path):
    with open(path,'r') as f:
        focals = [line.strip() for line in map(str,f)]
    return np.array(focals)    


# In[4]:


def save_image(images_rgb, path):
    images_bgr = cv2.cvtColor(images_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, images_bgr)


# ## Cylinder Warping

# In[5]:


def cylinder_warping(imgs, focals):
    cnt, h, w, c = imgs.shape
    warped_imgs = np.zeros([cnt, h, w, c])
    x_old = np.floor(w/2)
    y_old = np.floor(h/2)
    for i in range(cnt):
        for y_new in range(h):
            for x_new in range(w):
                s = float(focals[i])
                x = s * np.tan((x_new+1-x_old)/s)
                y = np.sqrt(x**2 + s**2)*(y_new+1-y_old)/s
                x = x + x_old
                y = y + y_old
                x = int(np.round(x))
                y = int(np.round(y))
                if x > 0 and x <= w and y > 0 and y <= h:
                    warped_imgs[i, y_new, x_new, :] = imgs[i, y-1, x-1, :]
                else:
                    warped_imgs[i, y_new, x_new, :] = 0
        

    return warped_imgs.astype(np.uint8)  


# ## Features (Detection + Descibtor + Matching)

# In[6]:


def Compute_Response(img,kernel=5,sigma=3,k=0.04):
    K = (kernel,kernel)

    img_blur = cv2.GaussianBlur(img,K,sigma)
    Iy, Ix = np.gradient(img_blur)
    Ix_s = Ix ** 2
    Iy_s = Iy ** 2
    Ixy = Ix * Iy

    Sx_s = cv2.GaussianBlur(Ix_s,K,sigma)
    Sy_s = cv2.GaussianBlur(Iy_s,K,sigma)
    Sxy = cv2.GaussianBlur(Ixy,K,sigma)

    detM = (Sx_s * Sy_s) - (Sxy ** 2)
    traceM = Sx_s + Sy_s

    R = detM - k * (traceM ** 2)
    print('Rmax:', np.max(R), 'Rmin:', np.min(R))
    return R, Ix, Iy, Ix_s, Iy_s


# In[7]:


def get_local_max_R(R,rthres=0.06):
    
    ker = np.ones((3,3), np.uint8)
    ker[1,1] = 0
    localMax = np.ones(R.shape, dtype=np.uint8)
    if np.max(R) > 600000:
        localMax[R <= np.max(R) * 0.03] = 0
    else:    
        localMax[R <= np.max(R) * rthres] = 0
    R_dila = cv2.dilate(R,ker)
    
    for i in range(localMax.shape[0]):
        for j in range(localMax.shape[1]):
            if localMax[i, j] == 1 and R[i, j] > R_dila[i, j]:
                localMax[i, j] = 1
            else:
                localMax[i, j] = 0

    print('corner nums:', np.sum(localMax))
    feature_points = np.where(localMax > 0)
    return feature_points[0], feature_points[1]   


# In[8]:


def orientation(Ix, Iy, Ix_s, Iy_s, fpy, fpx, bins=36, ksize=9):
    kernel9 = (ksize,ksize)
    
    M = (Ix_s + Iy_s) ** (1/2)
    theta = np.arctan(Iy / (Ix + 1e-8)) * (180 / np.pi)
    theta[Ix < 0] += 180
    theta = (theta + 360) % 360
    
    M_pad = cv2.copyMakeBorder(M,4,4,4,4, cv2.BORDER_CONSTANT, value=0)
    theta_pad = cv2.copyMakeBorder(theta,4,4,4,4, cv2.BORDER_CONSTANT, value=0)
    
    ori = np.ones((len(fpx),2))*-1
    mag = np.ones((len(fpx),2))*-1
    
    for i in range(len(fpx)):
        vote = np.zeros((bins,1))
        window = M_pad[int(fpy[i]+4-np.floor(ksize/2)):int(fpy[i]+4+np.floor(ksize/2)+1),int(fpx[i]+4-np.floor(ksize/2)):int(fpx[i]+4+np.floor(ksize/2)+1)]
        window_scaled = cv2.GaussianBlur(window,kernel9,1.5)
        window_t = theta_pad[int(fpy[i]+4-np.floor(ksize/2)):int(fpy[i]+4+np.floor(ksize/2)+1),int(fpx[i]+4-np.floor(ksize/2)):int(fpx[i]+4+np.floor(ksize/2)+1)]
        for m in range(window_t.shape[0]):
            for n in range(window_t.shape[1]):
                vote[int(window_t[m, n] // 10)] += window_scaled[m, n]
        vote_fake = vote        
        ind_1, mag_1 = np.argmax(vote), np.max(vote)
        vote_fake[ind_1] = np.min(vote)
        ind_2, mag_2 = np.argmax(vote_fake), np.max(vote_fake)
        if mag_2 >= mag_1 * 0.8:
            ori[i,0], ori[i,1] = ind_1, ind_2
            mag[i,0], mag[i,1] = mag_1, mag_2
        else:
            ori[i,0] = ind_1
            mag[i,0] = mag_1
    
    return ori, mag, M, theta


# In[10]:


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# In[11]:


def descriptor(fpx, fpy, theta, M):
    bins = 8
    left_descriptors = []
    right_descriptors = []
    h,w = M.shape
    middle = w/2
    real_corner = 0
    for i in range(len(fpy)):
        if fpx[i]-15 > 0 and fpx[i]+15 < w and fpy[i]-15 > 0 and fpy[i]+15 < h:
            real_corner += 1
            desc = []
            fea_block = theta[int(fpy[i]-7):int(fpy[i]+8)+1,int(fpx[i]-7):int(fpx[i]+8)+1]
            M_block = M[int(fpy[i]-7):int(fpy[i]+8)+1,int(fpx[i]-7):int(fpx[i]+8)+1]
            for m in range(0,fea_block.shape[0],4):
                for n in range(0,fea_block.shape[1],4):
                    window_M = M_block[m:m+4,n:n+4]
                    window_theta = fea_block[m:m+4,n:n+4]
                    vote_his = np.zeros((bins))
                    for o in range(window_M.shape[0]):
                        for p in range(window_M.shape[1]):
                            vote_his[int(window_theta[o,p] // 45)] += window_M[o, p]  
                    desc += vote_his.tolist()
            desc = normalize(desc)
            if np.any(desc>0.2):
                desc[desc>0.2] = 0.2
                desc = normalize(desc)
            if fpx[i] >= middle:
                right_descriptors.append({'point':(fpy[i], fpx[i]), 'desc':desc})
            else:
                left_descriptors.append({'point':(fpy[i], fpx[i]), 'desc':desc})
    print('realcorner:',real_corner)        
    return left_descriptors, right_descriptors                       


# In[12]:


def find_matches(des1, des2, thres=0.8):
    data_frame1 = pd.DataFrame(des1)
    data_frame2 = pd.DataFrame(des2)
    
    desc1 = data_frame1.loc[:]['desc'].tolist()
    desc2 = data_frame2.loc[:]['desc'].tolist()
    
    distances = cdist(desc1, desc2)
    sorted_index = np.argsort(distances, axis=1)
    matches = []
    for i, si in enumerate(sorted_index):
        first_match = distances[i, si[0]]
        second_match = distances[i, si[1]]
        if (first_match / second_match) < thres:
            matches.append([i, si[0]])
    
    print('matches nums:', len(matches))
    return matches


# ## Ransac

# In[13]:


def ransac(matches, des1, des2, n=1, K=1000):
    matches = np.array(matches)
    m1, m2 = matches[:, 0], matches[:, 1]
    
    df1 = pd.DataFrame(des1)
    df2 = pd.DataFrame(des2)
    
    P1_pre = np.array(df1.loc[m1][['point']])
    P2_pre = np.array(df2.loc[m2][['point']])
    
    
    P1, P2 = [[0, 0]], [[0, 0]]
    
    for i in P1_pre:
        P1 = np.concatenate((P1, np.array([[i[0][1], i[0][0]]])), axis=0)
    for i in P2_pre:                        
        P2 = np.concatenate((P2, np.array([[i[0][1], i[0][0]]])), axis=0)
        
    P1 = np.delete(P1, 0, 0)
    P2 = np.delete(P2, 0, 0)

    Err, Dxy = [], []
    for k in range(K):
        samples = np.random.randint(0, len(P1), n)
        dxy = np.mean(P1[samples] - P2[samples], axis=0).astype(np.int)
        diff_xy = np.abs(P1 - (P2 + dxy))
        err = np.sum(np.sign(np.sum(diff_xy, axis=1)))
        Err += [err]
        Dxy += [dxy]

    Einx = np.argsort(Err)
    best_dxy = np.round(Dxy[Einx[0]]).astype(int)
    
    return best_dxy


# ## Image Blending: Linear

# In[14]:


def new_panorama(Dxy_all, image_shape):
    print('[new panorama]')
    h, w, c = image_shape
    
    Dx, Dy = Dxy_all[:, 0], Dxy_all[:, 1]
    dx_max, dx_min = np.max(Dx), np.min(Dx)
    dy_max, dy_min = np.max(Dy), np.min(Dy)
    
    offset_x = -dx_min if dx_min < 0 else 0
    offset_y = -dy_min if dy_min < 0 else 0
    
    pano_w = (offset_x + dx_max + w) if dx_max > 0 else w + offset_x
    pano_h = (offset_y + dy_max + h) if dy_max > 0 else h + offset_y
    
    pano = np.zeros((pano_h, pano_w, c)).astype(np.float32)
    
    return pano, offset_x, offset_y


# In[15]:


def blend_linear_helper(pano, im, x, y, sign):
    h, w, c = im.shape
    if np.sum(pano) == 0:
        pano[y:y+h, x:x+w] = im.astype(np.float32)
    else:
        w_pano = np.sign(pano)
    
        w_blend = np.zeros(pano.shape)
        w_blend[y:y+h, x:x+w][im > 0] = 1
        
        union = np.sign(w_pano + w_blend)
        array = w_blend + w_pano - union
        
        sum_x = np.sum(array, axis=0)
        sum_y = np.sum(array, axis=1)

        index_x = np.where(sum_x > 0)[0]
        start_x = index_x[0]
        end_x = index_x[-1] + 1
        index_y = np.where(sum_y > 0)[0]
        start_y = index_y[0]
        end_y = index_y[-1] + 1

        xlen = end_x-start_x
        ylen = end_y-start_y
        
        inter = np.zeros((ylen, xlen))
        if sign >= 0:
            inter += np.linspace(0, 1, xlen)
        else:
            inter += np.linspace(1, 0, xlen)
            
        inter = np.stack([inter, inter, inter], axis=2)
        
        add = np.zeros(pano.shape).astype(np.float32)
        add[y:y+h, x:x+w] = im.astype(np.float32)
        
        # blending
        w_blend[start_y:end_y, start_x:end_x] *= inter
        w_blend[pano == 0] = 1
        w_blend[add == 0] = 0
        
        w_pano[start_y:end_y, start_x:end_x] *= (1. - inter)
        w_pano[add == 0] = 1
        w_pano[pano == 0] = 0
        
        pano = w_pano * pano + w_blend * add
        
    return pano


# In[16]:


def blend_linear(images, Dxy):
    print('[start blending]')
    h, w, c = images[0].shape
    Dxy_all = [np.zeros(2).astype(int)]
    for dxy in Dxy:
        Dxy_all.append(Dxy_all[-1] + dxy)
    Dxy_all = np.array(Dxy_all)
    
    pano, offset_x, offset_y = new_panorama(Dxy_all, (h, w, c))
    Dxy = [np.zeros(2)] + Dxy
    
    for i, (image, dxy_all, dxy) in enumerate(zip(images, Dxy_all, Dxy)):
        dx_all, dy_all = dxy_all
        
        new_x = offset_x + dx_all
        new_y = offset_y + dy_all
        pano = blend_linear_helper(pano, image, new_x, new_y, dxy[0])
        
    return pano.astype(np.uint8)


# ## Bundle Adjustment

# In[17]:


def drift(pano):
    print('[start adjustment]')
    pano_gray = cv2.cvtColor(pano, cv2.COLOR_RGB2GRAY)
    h, w = pano_gray.shape
    
    sum_x = np.sum(pano_gray, axis=0)
    sum_y = np.sum(pano_gray, axis=1) 
    
    index_x = np.where(sum_x > 0)[0]
    start_x = index_x[0]
    end_x = index_x[-1] + 1
    
    index_y = np.where(sum_y > 0)[0]
    start_y = index_y[0]
    end_y = index_y[-1] + 1
    
    lc = pano_gray[:, start_x]
    ly = np.where(lc > 0)[0]
    u_left = [start_x, ly[0]]
    b_left = [start_x, ly[-1]]
    
    end_x -= 1
    rc = pano_gray[:, end_x]
    ry = np.where(rc > 0)[0]
    u_right = [end_x, ry[0]]
    b_right = [end_x, ry[-1]]
    
    c1 = np.float32([u_left, u_right, b_left, b_right])
    c2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    M = cv2.getPerspectiveTransform(c1, c2)
    pano_drift = cv2.warpPerspective(pano, M, (w, h))
    
    return pano_drift







def main():
# In[18]:

    dirname = '../data'
    filename = 'focal.txt'
    img_focal = read_focal(f'{dirname}/{filename}')
    images = read_img(dirname)

    print(images.shape)


    # In[19]:


    warp_imgs = cylinder_warping(images,img_focal)
    l_Des = []
    r_Des = []
    for i in range(len(warp_imgs)):
        print(i)
        w_imgs_gray = cv2.cvtColor(warp_imgs[i], cv2.COLOR_RGB2GRAY)
        R, Ix, Iy, Ix_s, Iy_s = Compute_Response(w_imgs_gray)
        fpy, fpx = get_local_max_R(R)
        ori, mag, M, theta = orientation(Ix, Iy, Ix_s, Iy_s, fpy, fpx)
        l_des, r_des = descriptor(fpx, fpy, theta, M)
        l_Des += [l_des]
        r_Des += [r_des]


    # In[20]:

    Dxy = []
    for i in range(len(warp_imgs)-1):
        print(i)
        matches = find_matches(l_Des[i], r_Des[i+1])
        Dxy += [ransac(matches, l_Des[i], r_Des[i+1])]


    # In[21]:

    pano = blend_linear(warp_imgs, Dxy)


    # In[22]:

    pano_final = drift(pano)
    save_image(pano_final, dirname[:] + '-panorama.jpg')

if __name__ == "__main__":
    main()