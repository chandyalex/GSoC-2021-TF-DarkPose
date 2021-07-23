# Darkpose Tensorflow
Dark Pose implementation Tensorflow

Link to TF Model Garden GitHub issue: https://github.com/tensorflow/models/issues/8713 \
Link to paper: https://arxiv.org/abs/1910.06278

### Objective
Accurate coordinate representation in human pose estimation is big challenging problem
in computer vision, various deep learning methods are used to solve this problem and the
heat map plays a key role in the performance and accuracy of the model. However the key
role of heat map is not investigated in the most of the research papers,Distribution-Aware
Coordinate Representation for Human Pose Estimation mainly focused on the key role
of heat map in identifying the key points of human pose estimation. This work has great
potential in deep learning domain and could be able to produce revolutionising future works
like YOLO  did in the past few years. That’s the main reason why tensor flow community need
This model is implemented in TensorFlow 2 and makes it available for the open source community.This can be used for research purpose and industry in the future and has a greater potential for producing state of the art deep learning models using TensorFlow.

### Project status

- [x]  Data preparation
- [x]  Data generator
- [x]  Base line model
- [x]  Loss function
- [x]  Evaluation matrix
- [ ]  Hour glass model
- [ ]  HR Net
- [ ]  Dark pose implementation
- [x]  Training script base model https://github.com/chandyalex/Darkpose_Tensorflow/blob/main/jupyter/keras_test.ipynb
- [ ]  Training of base model with darpose
- [ ]  Training with pre-trained model
- [ ]  Proper configuration
- [ ]  Detailed Read me
- [ ]  Coding standard check
- [ ]  Launch

### Requirements
Install python packages using requirement.txt

### Results

| Model  | Accuracy | Download link|Colab note to train |
| ------------- | ------------- |-------------|-------------|
|  | | ||
|   |  | ||

# How to set up repository

#### Data set Download

#### Config file

#### Train model

#### Save model

#### Inference

#### How Use dark pose in other models

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
