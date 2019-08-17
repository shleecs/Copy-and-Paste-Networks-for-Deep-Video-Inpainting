# Copy-and-Paste-Networks-for-Deep-Video-Inpainting (ICCV 2019)
Official pytorch implementation for "Copy-and-Paste Networks for Deep Video Inpainting" (ICCV 2019) V.0.2

## Sungho Lee, [Seoung Wug Oh](https://sites.google.com/view/seoungwugoh), DaeYeun Won, [Seon Joo Kim](https://sites.google.com/site/seonjookim/)

[Paper] [[Video results](https://youtu.be/BKdxR9bQQMU)]

### Requirements
- python 3.6
- pytorch 0.4.0

### Download weights
```
mkdir ./weight
wget -O ./weight/weight.pth "https://drive.google.com/file/d/1tUeoalZ7J_4xWyJSw_cGvcJtZ9PJ-Nwy/view?usp=sharing"
```

### Run
```
python CPNet_test.py -g [gpu_num] -D [dataset_path]
```

### Use
This software is for non-commercial use only.

If you use this code please cite:

```
@InProceedings{lee2019cpnet,
author = {Lee, Sungho and Oh, Seoung Wug and Won, DaeYeun and Kim, Seon Joo},
title = {Copy-and-Paste Networks for Deep Video Inpainting},
booktitle = {International Conference on Computer Vision (ICCV)},
year = {2019}
}
```
