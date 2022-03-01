# VFX Project 2 - Image Stitching

## 0. Team Members
* R09922131 鄒汶橙
* R09922136 廖婕吟

## 1. Program Usage

### 1.1. Quick Usage

* Run
  
```
python3 image_stitching.py
```
 

## 2. Inputs and parameters

### 2.1. Input photos
* Input Images - `data`

### 2.2. Parameters

| Function | Parameter | Value |
| -------- | -------- | -------- |
| Compute_Response | kernel | 5 |
|                  | sigma  | 3 |
|                  | k    | 0.04|
| get_local_max_R  | rthres | 0.06|
| orientation      | ksize  | 9 |
| descriptor       | up     | 15 |
|                  | down   | 15 |
|                  | left   | 15 |
|                  | right  | 15 |
| ransac     | n     | 1        |
|            | K     | 1000     |
