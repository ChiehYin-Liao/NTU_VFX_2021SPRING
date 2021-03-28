# VFX Project 1 - High Dynamic Range Imaging

## 0. Team Members
* R09922131 鄒汶橙
* R09922136 廖婕吟

## 1. Program Usage

### 1.1. Quick Usage

* Run
  
```
python3 hdr+tonemap.py
```


## 2. Algorithms

### 2.1. Image Alignment

* MTB Algorithm 

### 2.2. HDR Reconstruction

* Debevec's Method

### 2.3. Tone Mapping

* Photographic

  * Global Operator
  * Local Operator
 

## 3. Inputs and parameters

### 3.1. Input photos
* Input Images - `data/original photos`
* Exposure Times
```
IMG_0071    1/800  
IMG_0072    1/640   
IMG_0074    1/400  
IMG_0075    1/320  
IMG_0077    1/200   
IMG_0078    1/160   
IMG_0080    1/100  
IMG_0081    1/80   
IMG_0083    1/50    
IMG_0084    1/40    
IMG_0086    1/25    
IMG_0087    1/20    
IMG_0089    1/13    
IMG_0090    1/10    
IMG_0092    1/6     
IMG_0093    1/5 
```

### 3.2. Parameter

* Image Alignment

  * threshold = $4$
  * depth = $4$

* Debevec's Method

  * $\lambda = 5$

* Photographic Global Operator

  * $\delta=10^{-6}$
  * $a=0.5$

* Photographic Local Operator
  
  * $S_{max}= 25$
  * $a=1.0$
  * $\phi= 8.0$
  * $\epsilon=0.01$
  