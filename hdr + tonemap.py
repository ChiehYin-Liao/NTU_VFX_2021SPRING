import numpy as np
import cv2
from numpy import linalg as la
import math
import random
import os.path as osp
import os


# weighting function value for pixel value z
def wf(z):
    what = 0
    zmin = 0
    zmax = 255
    if z <= (zmin + zmax) / 2:
        what = z - zmin
    else:
        what = zmax - z

    return what


# response curve recovery
def gsolve(Z, B, l, w):
    # initialize
    n = 256
    g = np.zeros(n)
    lE = np.zeros(Z.shape[0])
    A = np.zeros((Z.shape[0] * Z.shape[1] + n + 1, n + Z.shape[0]))
    b = np.zeros((A.shape[0], 1))

    # data-fitting equation
    k = 0
    for i in range(0, Z.shape[0]):
        for j in range(0, Z.shape[1]):
            wij = w[Z[i, j]]
            A[k, Z[i, j]] = wij
            A[k, n + i] = -wij
            b[k] = wij * B[j]
            k = k + 1

    # fix the curve by setting its middle value to 0
    A[k, 128] = 1
    k = k + 1

    # smoothness
    for i in range(0, n - 2):
        A[k, i] = l * w[i + 1]
        A[k, i + 1] = -2 * l * w[i + 1]
        A[k, i + 2] = l * w[i + 1]
        k = k + 1

    # U, S, V = la.svd(A)
    # Sigma = np.zeros((V.shape[0],U.shape[0]))
    # for i in range(0, Sigma.shape[0]):
    #     for j in range(0, Sigma.shape[1]):
    #         Sigma[i,i] = S[i]
    #
    # x = np.transpose(V).dot(Sigma).dot(np.transpose(U)).dot(b)
    A_inv = np.linalg.pinv(A)
    x = np.dot(A_inv, b)
    g = x[0:n]
    lE = x[n:x.shape[0]]

    return g, lE


def photographic_global(R, d, a):
    Lw = R
    Lw_bar = np.exp(np.mean(np.log(d + Lw)))
    Lm = (a / Lw_bar) * Lw
    Lwhite = np.max(Lm)
    Ld = (Lm * (1 + (Lm / Lwhite ** 2))) / (1 + Lm)
    pg_hdr = np.clip(np.array(Ld * 255), 0, 255).astype(np.uint8)

    cv2.imwrite("tonemap_photographic_global_lib1.jpg", pg_hdr)
    return pg_hdr


def gaussian_blur(img, smax=25, a=1.0, fi=8.0, e=0.01):
    m = img.shape[0]
    n = img.shape[1]
    blur_pre = img
    num = int((smax + 1) / 2)

    blur_list = np.zeros(img.shape + (num,))
    Vs_list = np.zeros(img.shape + (num,))

    for i, s in enumerate(range(1, smax + 1, 2)):
        blur = cv2.GaussianBlur(img, (s, s), 0)
        Vs = np.abs((blur - blur_pre) / (2 ** fi * a / s ** 2 + blur_pre))
        blur_list[:, :, i] = blur
        Vs_list[:, :, i] = Vs

        smax = np.argmax(Vs_list > e, axis=2)
        smax[np.where(smax == 0)] = 1
        smax -= 1

        I, J = np.ogrid[:m, :n]
        blur_smax = blur_list[I, J, smax]

        return blur_smax


def photographic_local(R, d=1e-6, a=0.5):
    pl_hdr = np.zeros(R.shape,dtype=np.float32)
    for channel in range(3):
        Lw = R[:,:,channel]
        Lw_bar = np.exp(np.mean(np.log(d + Lw)))
        Lm = (a / Lw_bar) * Lw
        Ls = gaussian_blur(Lm)
        Ld = Lm / (1 + Ls)
        pl_hdr[:,:,channel] = np.clip(np.array(Ld * 255), 0, 255).astype(np.uint8)

    cv2.imwrite("tonemap_photographic_local_lib1.jpg", pl_hdr)

    return pl_hdr


def main():

    dirname = 'photo_library1'
    imgs = []

    for filename in np.sort(os.listdir(dirname)):
        if osp.splitext(filename)[1] in ['.jpg', '.png', '.JPG']:
            img = cv2.imread(osp.join(dirname,filename))
            imgs += [img]

    img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12, img13, img14, img15, img16 = imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], imgs[5], imgs[6], imgs[7], imgs[8], imgs[9], imgs[10], imgs[11], imgs[12], imgs[13], imgs[14], imgs[15]

    img1 = cv2.pyrDown(cv2.pyrDown(img1))
    img2 = cv2.pyrDown(cv2.pyrDown(img2))
    img3 = cv2.pyrDown(cv2.pyrDown(img3))
    img4 = cv2.pyrDown(cv2.pyrDown(img4))
    img5 = cv2.pyrDown(cv2.pyrDown(img5))
    img6 = cv2.pyrDown(cv2.pyrDown(img6))
    img7 = cv2.pyrDown(cv2.pyrDown(img7))
    img8 = cv2.pyrDown(cv2.pyrDown(img8))
    img9 = cv2.pyrDown(cv2.pyrDown(img9))
    img10 = cv2.pyrDown(cv2.pyrDown(img10))
    img11 = cv2.pyrDown(cv2.pyrDown(img11))
    img12 = cv2.pyrDown(cv2.pyrDown(img12))
    img13 = cv2.pyrDown(cv2.pyrDown(img13))
    img14 = cv2.pyrDown(cv2.pyrDown(img14))
    img15 = cv2.pyrDown(cv2.pyrDown(img15))
    img16 = cv2.pyrDown(cv2.pyrDown(img16))


    random.seed(7414)
    sample = 50
    indices = random.sample(range(img1.shape[0] * img1.shape[1]), sample)
    index_i = np.zeros(sample, dtype=int)
    index_j = np.zeros(sample, dtype=int)
    for i in range(0, sample):
        index_i[i] = indices[i] // img1.shape[1]
        index_j[i] = indices[i] % img1.shape[1]

    ZB = np.zeros((sample, 16), dtype=int)
    ZG = np.zeros((sample, 16), dtype=int)
    ZR = np.zeros((sample, 16), dtype=int)
    for i in range(0, len(index_i)):
        ZB[i, 0] = int(img1[index_i[i], index_j[i], 0])
        ZB[i, 1] = int(img2[index_i[i], index_j[i], 0])
        ZB[i, 2] = int(img3[index_i[i], index_j[i], 0])
        ZB[i, 3] = int(img4[index_i[i], index_j[i], 0])
        ZB[i, 4] = int(img5[index_i[i], index_j[i], 0])
        ZB[i, 5] = int(img6[index_i[i], index_j[i], 0])
        ZB[i, 6] = int(img7[index_i[i], index_j[i], 0])
        ZB[i, 7] = int(img8[index_i[i], index_j[i], 0])
        ZB[i, 8] = int(img9[index_i[i], index_j[i], 0])
        ZB[i, 9] = int(img10[index_i[i], index_j[i], 0])
        ZB[i, 10] = int(img11[index_i[i], index_j[i], 0])
        ZB[i, 11] = int(img12[index_i[i], index_j[i], 0])
        ZB[i, 12] = int(img13[index_i[i], index_j[i], 0])
        ZB[i, 13] = int(img14[index_i[i], index_j[i], 0])
        ZB[i, 14] = int(img15[index_i[i], index_j[i], 0])
        ZB[i, 15] = int(img16[index_i[i], index_j[i], 0])
        ZG[i, 0] = int(img1[index_i[i], index_j[i], 1])
        ZG[i, 1] = int(img2[index_i[i], index_j[i], 1])
        ZG[i, 2] = int(img3[index_i[i], index_j[i], 1])
        ZG[i, 3] = int(img4[index_i[i], index_j[i], 1])
        ZG[i, 4] = int(img5[index_i[i], index_j[i], 1])
        ZG[i, 5] = int(img6[index_i[i], index_j[i], 1])
        ZG[i, 6] = int(img7[index_i[i], index_j[i], 1])
        ZG[i, 7] = int(img8[index_i[i], index_j[i], 1])
        ZG[i, 8] = int(img9[index_i[i], index_j[i], 1])
        ZG[i, 9] = int(img10[index_i[i], index_j[i], 1])
        ZG[i, 10] = int(img11[index_i[i], index_j[i], 1])
        ZG[i, 11] = int(img12[index_i[i], index_j[i], 1])
        ZG[i, 12] = int(img13[index_i[i], index_j[i], 1])
        ZG[i, 13] = int(img14[index_i[i], index_j[i], 1])
        ZG[i, 14] = int(img15[index_i[i], index_j[i], 1])
        ZG[i, 15] = int(img16[index_i[i], index_j[i], 1])
        ZR[i, 0] = int(img1[index_i[i], index_j[i], 2])
        ZR[i, 1] = int(img2[index_i[i], index_j[i], 2])
        ZR[i, 2] = int(img3[index_i[i], index_j[i], 2])
        ZR[i, 3] = int(img4[index_i[i], index_j[i], 2])
        ZR[i, 4] = int(img5[index_i[i], index_j[i], 2])
        ZR[i, 5] = int(img6[index_i[i], index_j[i], 2])
        ZR[i, 6] = int(img7[index_i[i], index_j[i], 2])
        ZR[i, 7] = int(img8[index_i[i], index_j[i], 2])
        ZR[i, 8] = int(img9[index_i[i], index_j[i], 2])
        ZR[i, 9] = int(img10[index_i[i], index_j[i], 2])
        ZR[i, 10] = int(img11[index_i[i], index_j[i], 2])
        ZR[i, 11] = int(img12[index_i[i], index_j[i], 2])
        ZR[i, 12] = int(img13[index_i[i], index_j[i], 2])
        ZR[i, 13] = int(img14[index_i[i], index_j[i], 2])
        ZR[i, 14] = int(img15[index_i[i], index_j[i], 2])
        ZR[i, 15] = int(img16[index_i[i], index_j[i], 2])

    B = np.zeros(16)
    splib2 = np.array([1/800,1/640,1/400,1/320,1/200,1/160,1/100,1/80,1/50,1/40,1/25,1/20,1/13,1/10,1/6,1/5])
    splib1 = np.array([1/4,1/5,1/8,1/10,1/15,1/20,1/30,1/50,1/60,1/80,1/125,1/160,1/250,1/320,1/500,1/640])
    sphall1 = np.array([1/1000,1/640,1/400,1/250,1/160,1/100,1/80,1/60,1/40,1/25,1/15,1/10,1/6,1/4,0.4,0.6])
    if dirname == "photo_hallway1" or dirname == "aligned_hallway1":
        for i in range(16):
            B[i] = np.log(sphall1[i])
    elif dirname == "photo_library1" or dirname == "aligned_library1":
        for i in range(16):
            B[i] = np.log(splib1[i])
    else:
        for i in range(16):
            B[i] = np.log(splib2[i])

    l = 5
    w = np.zeros(256)
    for i in range(0, 256):
        w[i] = wf(i)

    gB, LEB = gsolve(ZB, B, l, w)
    gG, LEG = gsolve(ZG, B, l, w)
    gR, LER = gsolve(ZR, B, l, w)
    HDRimg = np.zeros(img1.shape)
    for i in range(0, HDRimg.shape[0]):
        for j in range(0, HDRimg.shape[1]):
            lnEB = (wf(img1[i, j, 0]) * (gB[img1[i, j, 0]] - B[0]) + wf(img2[i, j, 0]) * (
                        gB[img2[i, j, 0]] - B[1]) + wf(img3[i, j, 0]) * (gB[img3[i, j, 0]] - B[2]) + wf(
                img4[i, j, 0]) * (gB[img4[i, j, 0]] - B[3]) + \
                    wf(img5[i, j, 0]) * (gB[img5[i, j, 0]] - B[4]) + wf(img6[i, j, 0]) * (
                                gB[img6[i, j, 0]] - B[5]) + wf(img7[i, j, 0]) * (gB[img7[i, j, 0]] - B[6]) + wf(
                        img8[i, j, 0]) * (gB[img8[i, j, 0]] - B[7]) + \
                    wf(img9[i, j, 0]) * (gB[img9[i, j, 0]] - B[8]) + wf(img10[i, j, 0]) * (
                                gB[img10[i, j, 0]] - B[9]) + wf(img11[i, j, 0]) * (gB[img11[i, j, 0]] - B[10]) + wf(
                        img12[i, j, 0]) * (gB[img12[i, j, 0]] - B[11]) + \
                    wf(img13[i, j, 0]) * (gB[img13[i, j, 0]] - B[12]) + wf(img14[i, j, 0]) * (
                                gB[img14[i, j, 0]] - B[13]) + wf(img15[i, j, 0]) * (gB[img15[i, j, 0]] - B[14]) + wf(
                        img16[i, j, 0]) * (gB[img16[i, j, 0]] - B[15])) \
                   / (wf(img1[i, j, 0]) + wf(img2[i, j, 0]) + wf(img3[i, j, 0]) + wf(img4[i, j, 0]) + wf(
                img5[i, j, 0]) + wf(img6[i, j, 0]) + wf(img7[i, j, 0]) + wf(img8[i, j, 0]) + \
                      wf(img9[i, j, 0]) + wf(img10[i, j, 0]) + wf(img11[i, j, 0]) + wf(img12[i, j, 0]) + wf(
                        img13[i, j, 0]) + wf(img14[i, j, 0]) + wf(img15[i, j, 0]) + wf(img16[i, j, 0]))

            HDRimg[i, j, 0] = math.exp(lnEB)
            lnEG = (wf(img1[i, j, 1]) * (gG[img1[i, j, 1]] - B[0]) + wf(img2[i, j, 1]) * (
                        gG[img2[i, j, 1]] - B[1]) + wf(img3[i, j, 1]) * (gG[img3[i, j, 1]] - B[2]) + wf(
                img4[i, j, 1]) * (gG[img4[i, j, 1]] - B[3]) + \
                    wf(img5[i, j, 1]) * (gG[img5[i, j, 1]] - B[4]) + wf(img6[i, j, 1]) * (
                                gG[img6[i, j, 1]] - B[5]) + wf(img7[i, j, 1]) * (gG[img7[i, j, 1]] - B[6]) + wf(
                        img8[i, j, 1]) * (gG[img8[i, j, 1]] - B[7]) + \
                    wf(img9[i, j, 1]) * (gG[img9[i, j, 1]] - B[8]) + wf(img10[i, j, 1]) * (
                                gG[img10[i, j, 1]] - B[9]) + wf(img11[i, j, 1]) * (gG[img11[i, j, 1]] - B[10]) + wf(
                        img12[i, j, 1]) * (gG[img12[i, j, 1]] - B[11]) + \
                    wf(img13[i, j, 1]) * (gG[img13[i, j, 1]] - B[12]) + wf(img14[i, j, 1]) * (
                                gG[img14[i, j, 1]] - B[13]) + wf(img15[i, j, 1]) * (gG[img15[i, j, 1]] - B[14]) + wf(
                        img16[i, j, 1]) * (gG[img16[i, j, 1]] - B[15])) \
                   / (wf(img1[i, j, 1]) + wf(img2[i, j, 1]) + wf(img3[i, j, 1]) + wf(img4[i, j, 1]) + wf(
                img5[i, j, 1]) + wf(img6[i, j, 1]) + wf(img7[i, j, 1]) + wf(img8[i, j, 1]) + \
                      wf(img9[i, j, 1]) + wf(img10[i, j, 1]) + wf(img11[i, j, 1]) + wf(img12[i, j, 1]) + wf(
                        img13[i, j, 1]) + wf(img14[i, j, 1]) + wf(img15[i, j, 1]) + wf(img16[i, j, 1]))

            HDRimg[i, j, 1] = math.exp(lnEG)
            lnER = (wf(img1[i, j, 2]) * (gR[img1[i, j, 2]] - B[0]) + wf(img2[i, j, 2]) * (
                        gR[img2[i, j, 2]] - B[1]) + wf(img3[i, j, 2]) * (gR[img3[i, j, 2]] - B[2]) + wf(
                img4[i, j, 2]) * (gR[img4[i, j, 2]] - B[3]) + \
                    wf(img5[i, j, 2]) * (gR[img5[i, j, 2]] - B[4]) + wf(img6[i, j, 2]) * (
                                gR[img6[i, j, 2]] - B[5]) + wf(img7[i, j, 2]) * (gR[img7[i, j, 2]] - B[6]) + wf(
                        img8[i, j, 2]) * (gR[img8[i, j, 2]] - B[7]) + \
                    wf(img9[i, j, 2]) * (gR[img9[i, j, 2]] - B[8]) + wf(img10[i, j, 2]) * (
                                gR[img10[i, j, 2]] - B[9]) + wf(img11[i, j, 2]) * (gR[img11[i, j, 2]] - B[10]) + wf(
                        img12[i, j, 2]) * (gR[img12[i, j, 2]] - B[11]) + \
                    wf(img13[i, j, 2]) * (gR[img13[i, j, 2]] - B[12]) + wf(img14[i, j, 2]) * (
                                gR[img14[i, j, 2]] - B[13]) + wf(img15[i, j, 2]) * (gR[img15[i, j, 2]] - B[14]) + wf(
                        img16[i, j, 2]) * (gR[img16[i, j, 2]] - B[15])) \
                   / (wf(img1[i, j, 2]) + wf(img2[i, j, 2]) + wf(img3[i, j, 2]) + wf(img4[i, j, 2]) + wf(
                img5[i, j, 2]) + wf(img6[i, j, 2]) + wf(img7[i, j, 2]) + wf(img8[i, j, 2]) + \
                      wf(img9[i, j, 2]) + wf(img10[i, j, 2]) + wf(img11[i, j, 2]) + wf(img12[i, j, 2]) + wf(
                        img13[i, j, 2]) + wf(img14[i, j, 2]) + wf(img15[i, j, 2]) + wf(img16[i, j, 2]))

            HDRimg[i, j, 2] = math.exp(lnER)

    cv2.imwrite("HDRlib1.hdr", HDRimg.astype(np.float32))
    photographic_global(HDRimg, 1e-6, 0.5)
    photographic_local(HDRimg)


if __name__ == '__main__':
    main()
