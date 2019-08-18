# Copy-and-Paste-Networks-for-Deep-Video-Inpainting (ICCV 2019)
Official pytorch implementation for "Copy-and-Paste Networks for Deep Video Inpainting" (ICCV 2019) V.0.2

#### Sungho Lee, [Seoung Wug Oh](https://sites.google.com/view/seoungwugoh), DaeYeun Won and [Seon Joo Kim](https://sites.google.com/site/seonjookim/)

[Paper]

### Video result
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/BKdxR9bQQMU/0.jpg)](https://www.youtube.com/watch?v=BKdxR9bQQMU)

### Requirements
- python 3.6
- pytorch 0.4

### Download weights
```
mkdir ./weight
wget -O ./weight/weight.pth "https://www.dropbox.com/s/vbh12ay2ubrw3m9/weight.pth?dl=0"
```

### Run
```
python CPNet_test.py -g [gpu_num] -D [dataset_path]
```

### Reference

If you use this code please cite:

```
@InProceedings{lee2019cpnet,
author = {Lee, Sungho and Oh, Seoung Wug and Won, DaeYeun and Kim, Seon Joo},
title = {Copy-and-Paste Networks for Deep Video Inpainting},
booktitle = {International Conference on Computer Vision (ICCV)},
year = {2019}
}
```

### Term of use

This software is for non-commercial use only.
The source code is released under the Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) Licence
(see [this](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) for details)
