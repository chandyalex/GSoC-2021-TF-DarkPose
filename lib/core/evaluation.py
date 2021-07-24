# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from core.inference_dark import get_max_preds


def calc_dists(preds, target, normalize):
    # preds= preds.numpy()
    # target = target.numpy()
    # normalize=normalize.numpy()
    preds = tf.cast(preds,np.float32)
    target = tf.cast(target ,np.float32)
    dist = tf.Variable(tf.zeros([preds.shape[1], preds.shape[0]]),np.float32)
    # dist = tf.Variable(dist)
    # dist = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            indices=[c,n]
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]

                normed_targets = target[n, c, :] / normalize[n]
                # normed_preds = tf.cast(normed_preds ,np.float32)
                # normed_targets = tf.cast(normed_targets ,np.float32)
                # normed_preds=normed_preds.numpy()
                # normed_targets = normed_targets.numpy()
                norm_values,norm_values1 = tf.linalg.normalize(tf.math.subtract(normed_preds, normed_targets))
                # norm_values=norm_values.numpy()
                # norm_values1=norm_values1.numpy()
                # print(norm_values1[0])
                # print(norm_values[0])
                dist=dist[c,n].assign(norm_values[0])

                # dists.assign(c, n) =
            else:
                dist=dist[c,n].assign(-1)

    return dist


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    # dist_acc=dist_acc.numpy()
    dist_cal = tf.math.not_equal(dists, -1)
    thr = tf.constant(thr,np.float32)

    num_dist_cal = tf.math.reduce_sum(tf.cast(dist_cal, tf.float32))
    if num_dist_cal > 0:
        less= tf.math.less(dists[dist_cal], thr)
        reduce_less=tf.math.reduce_sum(tf.cast(less, tf.float32))
        return reduce_less* 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    output = output.numpy()
    target = target.numpy()


    idx = list(range(output.shape[3]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[1]
        w = output.shape[2]
        norm = tf.math.multiply(tf.ones([pred.shape[0], 2]) , tf.Variable([h, w],dtype=np.float32)) / 10
    dists = calc_dists(pred, target, norm)

    acc = tf.Variable(tf.zeros([len(idx)]))
    avg_acc = 0
    cnt = 0


    for i in range(len(idx)):
        acc = acc[i].assign(dist_acc(dists[idx[i]]))
        if acc[i] >= 0:
            avg_acc = avg_acc + acc[i]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc = acc[0].assign(avg_acc)
    # arg_acc = tf.convert_to_tensor(acc,dtype=np.float32)
    # arg_avg_acc=tf.convert_to_tensor(avg_acc,dtype=np.float32)
    # arg_pred= tf.convert_to_tensor(pred,dtype=np.float32)
    # cnt=tf.convert_to_tensor(cnt,dtype=np.float32)
    # print(acc.shape)
    # print(avg_acc.shape)
    # print(pred.shape)

    # pred=pred.reshape(output.shape[0],2,-1)
    # print(pred.shape)
    # avg_acc, cnt, pred

    return acc