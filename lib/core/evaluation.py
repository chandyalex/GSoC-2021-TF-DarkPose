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

import numpy as np
import tensorflow as tf

from core.inference_dark import get_max_preds


def calc_dists(preds, target, normalize):

  preds = tf.cast(preds, np.float32)
  target = tf.cast(target, np.float32)
  dist = tf.Variable(tf.zeros([preds.shape[1], preds.shape[0]]), np.float32)
 
  for n in range(preds.shape[0]):
    for c in range(preds.shape[1]):
      indices = [c, n]
      if target[n, c, 0] > 1 and target[n, c, 1] > 1:
        normed_preds = preds[n, c, :] / normalize[n]
        normed_targets = target[n, c, :] / normalize[n]
        norm_values, norm_values1 = tf.linalg.normalize(
            tf.math.subtract(normed_preds, normed_targets)
            )
        dist = dist[c, n].assign(norm_values[0])

      else:

        dist = dist[c, n].assign(-1)

  return dist


def dist_acc(dists, thr=0.5):
  ''' Return percentage below threshold while ignoring values with a -1 '''

  dist_cal = tf.math.not_equal(dists, -1)
  thr = tf.constant(thr, np.float32)

  num_dist_cal = tf.math.reduce_sum(tf.cast(dist_cal, tf.float32))
  if num_dist_cal > 0:
    less = tf.math.less(dists[dist_cal], thr)
    reduce_less = tf.math.reduce_sum(tf.cast(less, tf.float32))
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
  if not isinstance(output, np.ndarray):
    output = output.numpy()
  if not isinstance(target, np.ndarray):
    target = target.numpy()

  idx = list(range(output.shape[3]))
  norm = 1.0
  if hm_type == 'gaussian':
    pred, _ = get_max_preds(output)
    target, _ = get_max_preds(target)
    h = output.shape[1]
    w = output.shape[2]
    norm = tf.math.multiply(
        tf.ones([pred.shape[0], 2]), tf.Variable([h, w], dtype=np.float32)) / 10
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


  return acc, avg_acc, cnt, pred
