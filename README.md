
# Video Matting via Consistency-Regularized Graph Neural Networks
### [Project Page]() | [Real Data](https://www.dropbox.com/sh/23uvsue5we7e7b5/AAB4GSSWIaKiSouvN3wuWiwWa?dl=0) | [Paper](https://faculty.ucmerced.edu/mhyang/papers/iccv2021_video_matting.pdf)


## Background

### Local env
1. Create virtual env: ```python3 -m venv crgnn```
2. Activate virtual env: ```source crgnn/bin/activate```

### Docker env

1. Run following command to build docker image:

```
    docker build -t . crgnn
```

## Installation
Our code has been tested on Python 3.8, cuda 11.4 and PyTorch 1.10.0.

	pip install -r requirements.txt
	# install dcn
	cd models/archs/dcn
	python setup.py develop

## Inference

### Local env
Run the following command to do inference of CRGNN on the video matting dataset:

    python3 test.py --data-root ./examples --checkpoint ./checkpoint/e20.pth

### Docker env

    docker run -it --rm --gpus=all --ipc=host crgnn bash


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
