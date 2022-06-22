
# Video Matting via Consistency-Regularized Graph Neural Networks
### [Project Page]() | [Real Data](https://www.dropbox.com/sh/23uvsue5we7e7b5/AAB4GSSWIaKiSouvN3wuWiwWa?dl=0) | [Paper](https://faculty.ucmerced.edu/mhyang/papers/iccv2021_video_matting.pdf)


## Background
1. Run following command to build docker image:

```
docker build -t . crgnn
```

Code has been tested on Python 3.8, cuda 11.4 and PyTorch 1.10.0.

## Inference

1. Run docker container using following command:
```
docker run -it --rm --gpus=all --ipc=host -v `pwd`/<data-folder>:/data crgnn bash
```

2. Run test.py script:
```
python3 test.py --data-root ./examples --checkpoint ./checkpoint/e20.pth
```



## Data
1. Please see the real data in the above link.
2. Please contact Tiantian Wang (tiantianwang.ice@gmail.com) if you need composited data.

## Citation
If you find this work or code useful for your research, please cite:
```
@inproceedings{wang2021crgnn,
  title={Video Matting via Consistency-Regularized Graph Neural Networks},
  author={Wang, Tiantian and Liu, Sifei and Tian, Yapeng and Li, Kai and Yang, Ming-Hsuan},
  booktitle={Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}

```

## Permission and Disclaimer
This code is only for non-commercial purposes. As covered by the ADOBE IMAGE DATASET LICENSE AGREEMENT, the trained models included in this repository can only be used/distributed for non-commercial purposes. Anyone who violates this rule will be at his/her own risk.
