# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com)
# Modified by Chandykunju Alex (chandyalex92@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random
import sys

import cv2
import numpy as np
import tensorflow as tf
# from tf.data import Dataset
#
import sys
sys.path.append("/home/chandy/gsoc/Darkpose_Tensorflow/lib")
#### to be removed later
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

dataset = tf.data.Dataset
