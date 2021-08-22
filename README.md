DISCLAIMER: this Darkpose Tensorflow implementation is still under development. No support will be provided during the development phase.

# Darkpose Tensorflow
Dark Pose implementation Tensorflow

This project is done as the part of google smmer of code 2021 

[GSOC 2021 project link](https://summerofcode.withgoogle.com/projects/#6367695945072640)

## Mentors

* Jaeyoun Kim [@Github jaeyounkim](https://github.com/jaeyounkim)
*  Margaret Maynard-Reid [@Github margaretmz](https://github.com/margaretmz)


<img src="assets/GSoC-icon-192.png" width="150"/> <img src="assets/TensorFlow_Brand/TensorFlow_Logo/Primary/PNG/TF_FullColor_Stacked.png" width="227"/> 

### Objective
Accurate coordinate representation in human pose estimation is big challenging problem
in computer vision, various deep learning methods are used to solve this problem and the
heat map plays a key role in the performance and accuracy of the model. However the key
role of heat map is not investigated in the most of the research papers,Distribution-Aware
Coordinate Representation for Human Pose Estimation mainly focused on the key role
of heat map in identifying the key points of human pose estimation. This work has great
potential in deep learning domain and could be able to produce revolutionising future works
like YOLO  did in the past few years. That’s the main reason why tensor flow community need
This approach is implemented in TensorFlow 2 as the part of GSoC-2021 and available for the open source community.This can be used for research purpose and industry in the future and has a greater potential for producing state of the art deep learning models using TensorFlow.

### Project status

- [x]  Data preparation
- [x]  Data generator
- [x]  Base line model
- [x]  Loss function
- [x]  Evaluation matrix
- [x]  Dark pose implementation
- [x]  [Training script base model](jupyter/keras_test.ipynb)
- [x]  [Training of base model with darkpose](/jupyter/Dark_pose_training_testing.ipynb)
- [ ]  Hour glass model
- [ ]  HR Net
- [ ]  Training with pre-trained model
- [x]  Proper configuration
- [x]  Detailed Read me
- [x]  Coding standard check
- [ ]  Launch

### Requirements
The resitory tested with \
Tensorflow 2.5 \
Anaconda 3.8




# How to set up repository

```
$ git clone https://github.com/chandyalex/GSoC-2021-TF-DarkPose
$ cd Darkpose_Tensorflow
$ pip install -r requirements.txt

```
Also install coco api using following command
```
git clone https://github.com/cocodataset/cocoapi.git

cd cocoapi/PythonAPI/

python setup.py install

```
After intalling all dendencies go to $REPO/lib
and run 
```
make
```
To make nms libraries.

Please note that nms library for TPU is still under development and you can find at the branch 'tpu-dark'
#### Data set Download

Download COCO dataset from here: https://cocodataset.org/#download

For the COCO dataset, your directory tree should look like this:

```
${POSE_ROOT}/data/coco
├── annotations
├── images
│   ├── test2017
│   ├── train2017
│   └── val2017
└── person_detection_results

```

#### Config file

To adjust Training parameters try to tweek the yaml file in /experiments folder

```
export CONFIG=../experiments/coco/resnet/res50_128x96_d256x3_adam_lr1e-3.yaml

export DATA_DIR="${POSE_ROOT}/data/coco"
export OUTPUT_DIR='home://repository/out"/'
export LOG_DIR='home://repository/log'
export ROOT="${POSE_ROOT}/data/coco"
export TEST_SET="val"
export TRAIN_SET="train"
```

#### Train model

* Train the model without dark 
```
# load the data
data_gen = coco.COCODataset(cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True)

# initialize model
model= pose_resnet.get_pose_net(cfg, is_train=True)
model.build(input_shape=(None,256, 256, 3))
model.compile(loss=JointsMSELoss, optimizer='SGD',run_eagerly=True,metrics=accuracy)
model.summary()

# train
model.fit(data_gen,epochs=2)
```
* Train the model with dark \
The dark method is implemented in the core library 
It can be aceesd [here](lib/core/dark_function.py)



```
from core.dark_function import train,validate

data_gen = coco.COCODataset(cfg, cfg.DATASET.ROOT, cfg.DATASET TRAIN_SET, True)
# Initialize model
model= pose_resnet.get_pose_net(cfg, is_train=True)
# Initilize optimizer
optimizer = get_optimizer(cfg, model)
#Start training uisng the loop method
epoch=2
for i in range(epoch):
    train(cfg,data_gen,model,JointsMSELoss,optimizer,i)
```

### Results

| Model  | Config name |
| ------------- | ------------- |
| **SimpleBaseline-R50 + DARK**  | `res50_128x96_d256x3_adam_lr1e-3.yaml` | 
| **SimpleBaseline-R101 + DARK** | `res101_128x96_d256x3_adam_lr1e-3.yaml` | 
| **SimpleBaseline-R152 + DARK** | `res152_128x96_d256x3_adam_lr1e-3.yaml` | 



## References

## Citation
If you use our code or models in your research, please cite with:
```
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {TPAMI}
  year={2019}
}

```
## Authors or Maintainers

* Chandykunju Alex ([@GitHub chandyalex](https://github.com/chandyalex))

