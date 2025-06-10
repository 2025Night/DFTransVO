# Neural Pose Estimation Based on Improved Detector-free Matching for Low-texture Environments

> Neural Pose Estimation Based on Improved Detector-free Matching for Low-texture Environments


## Installation
```shell
docker build -t dftransvo:1.0 .
docker run -it --user=root -v uploading_code/:/temp:rw --gpus --shm-size 32G –name build_env dftransvo:1.0
cd LoFTR
conda env create -f environment.yaml
conda activate dftransvo
pip install -r requirements.txt
exit
docker commit build_env dftransvo:2.0
```



## Test

```shell
docker run –rm -it --user=root -v uploading_code/:/temp:rw --gpus --shm-size 32G –name test dftransvo:2.0
python test_pair.py
```

## Acknowledgements
This project build upon the excellent work of: 
- [LoFTR](https://github.com/zju3dv/LoFTR)
- [EfficientLoFTR](https://github.com/zju3dv/efficientloftr)
