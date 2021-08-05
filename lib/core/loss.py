# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras.backend as K


#https://www.tensorflow.org/guide/keras/train_and_evaluate#custom_losses



def dice_loss(y_true, y_pred):

    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred))
    loss = 1. - 2 * intersection / (union + K.epsilon())
    print(loss)

    return loss
#https://www.mdpi.com/2072-4292/12/23/3857/pdf
#An Improved Deep Keypoint Detection Network
#for Space Targets Pose Estimation
def ohkm(loss,k=8):
    topk=k
    ohkm_loss = tf.constant(0, dtype=np.float32, shape=None, name='Const')
    sub_loss = loss

    topk_val, topk_idx = tf.math.top_k(
        sub_loss, k=topk,sorted=False
    )
    tmp_loss = tf.gather(sub_loss,topk_idx)
    ohkm_loss = tf.math.add(tmp_loss,ohkm_loss) / topk
    arg = tf.convert_to_tensor(ohkm_loss,dtype=np.float32)
    return arg

def JointsMSELoss(y_true, y_pred,target_weight=None):
    criterion = keras.losses.MeanSquaredError(reduction="auto")
    if not isinstance(y_pred,np.ndarray):
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        target_weight = target_weight.numpy()

    batch_size = y_pred.shape[0]
    num_joints = y_pred.shape[3]

    heatmaps_pred = tf.reshape(y_pred,(num_joints,batch_size, -1))

    # heatmaps_pred = np.split(heatmaps_pred , 1)
    heatmaps_gt = tf.reshape(y_true,(num_joints,batch_size, -1))
    # heatmaps_gt = np.split(heatmaps_gt , 1)
    loss = 0

    for idx in range(num_joints):


        heatmap_pred= tf.squeeze(heatmaps_pred[idx])
        heatmap_gt = tf.squeeze(heatmaps_gt[idx])
        if target_weight.any():
            loss = tf.math.add(loss, 0.5 * criterion(
                tf.math.multiply(heatmap_pred,target_weight[:, idx]),
                tf.math.multiply(heatmap_gt,target_weight[:, idx])))
        else:
            loss = tf.math.add(loss, 0.5 * criterion(heatmap_pred, heatmap_gt))

    loss_fin=loss/num_joints
    arg = tf.convert_to_tensor(loss_fin,dtype=np.float32)
    return arg

def JointsOHKMMSELoss(y_true, y_pred,target_weight=None):
    criterion = keras.losses.MeanSquaredError(reduction="auto")
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    batch_size = y_pred.shape[0]
    num_joints = y_pred.shape[3]

    heatmaps_pred = tf.reshape(y_pred,(num_joints,batch_size, -1))

    # heatmaps_pred = np.split(heatmaps_pred , 1)
    heatmaps_gt = tf.reshape(y_true,(num_joints,batch_size, -1))
    # heatmaps_gt = np.split(heatmaps_gt , 1)
    loss = []

    for idx in range(num_joints):

        heatmap_pred= tf.squeeze(heatmaps_pred[idx])
        heatmap_gt = tf.squeeze(heatmaps_gt[idx])
        if target_weight:
            target_weight = target_weight.numpy()
            loss.append(0.5 * criterion(
                heatmap_pred.dot(target_weight[:, idx]),
                heatmap_gt.dot(target_weight[:, idx]))
            )

        else:
            loss.append(0.5 * criterion(heatmap_pred, heatmap_gt))


    loss = [tf.math.reduce_mean(loss[l]) for l in range(len(loss))]

    loss = tf.concat(loss,axis=0)
    # loss_fin=[tf.math.reduce_mean(
    # l, axis=None, keepdims=False, name=None).unsqueeze(dim=1) for l in loss]
    # loss_fin = np.concatenate(loss_fin,axis=1)
    # print(loss_fin)

    return ohkm(loss)
