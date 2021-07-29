# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com) and Feng Zhang (zhangfengwcy@gmail.com)
#
#------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import numpy as np
import cv2


from utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'



    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[3]
    width = batch_heatmaps.shape[1]
    heatmaps_reshaped = tf.reshape(batch_heatmaps,(batch_size, num_joints, -1))
    idx = tf.math.argmax(heatmaps_reshaped, 2)
    maxvals = tf.math.reduce_max(heatmaps_reshaped, 2)

    maxvals = tf.reshape(maxvals,(batch_size, num_joints, 1))
    idx = tf.reshape(idx,(batch_size, num_joints, 1))

    preds = tf.Variable(tf.cast(tf.tile(idx, (1, 1, 2)),tf.float32))
    # preds=tf.cast(preds, tf.float32)

    preds = preds[:, :, 0].assign(tf.math.floormod(preds[:, :, 0], width))
    preds = preds[:, :, 1].assign(tf.math.floor((preds[:, :, 1]) / width))

    pred_mask = tf.tile(tf.math.greater(maxvals, 0.0),(1, 1, 2))
    pred_mask = tf.cast(pred_mask ,tf.float32)
    preds_new = tf.math.multiply(preds,pred_mask)



    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)
    if not isinstance(batch_heatmaps,np.ndarray):
        batch_heatmaps=batch_heatmaps.numpy()

    # heatmap_width = heatmap_width.numpy()
    coords=coords.numpy()
    maxvals=maxvals.numpy()

    heatmap_height = batch_heatmaps.shape[1]
    heatmap_width = batch_heatmaps.shape[2]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals
