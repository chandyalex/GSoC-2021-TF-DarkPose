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

import copy
import random
import logging

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence

from pycocotools.coco import COCO
from utils.transforms import fliplr_joints
from utils.transforms import affine_transform
from utils.transforms import get_affine_transform


dataset = tf.data.Dataset
logger = logging.getLogger(__name__)

class JointsDataset(Sequence):
  def __init__(self, cfg, root, image_set, is_train, transform=None):
    self.num_joints = 0
    self.pixel_std = 200
    self.flip_pairs = []
    self.parent_ids = []

    self.is_train = is_train
    self.root = root
    self.image_set = image_set
    self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU


    self.output_path = cfg.OUTPUT_DIR
    self.data_format = cfg.DATASET.DATA_FORMAT

    self.scale_factor = cfg.DATASET.SCALE_FACTOR
    self.rotation_factor = cfg.DATASET.ROT_FACTOR
    self.flip = cfg.DATASET.FLIP
    self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
    self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
    self.color_rgb = cfg.DATASET.COLOR_RGB

    self.target_type = cfg.MODEL.TARGET_TYPE
    self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
    # self.input_size = input_size
    # output heatmap size is 1/4 of input size
    self.output_size = (self.image_size[0]//4, self.image_size[1]//4)

    self.n_channels=3
    self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
    self.sigma = cfg.MODEL.SIGMA
    self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
    self.joints_weight = 1
    self.shuffle=True


    self.transform = transform
    self.db = []
    self.coco = copy.deepcopy(COCO(self._get_ann_file_keypoint()))
    self.image_set_index = copy.deepcopy(self._load_image_set_index())
    self.on_epoch_end()

  def _get_db(self):
    raise NotImplementedError

  def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
    raise NotImplementedError

  def half_body_transform(self, joints, joints_vis):
    upper_joints = []
    lower_joints = []
    for joint_id in range(self.num_joints):
      if joints_vis[joint_id][0] > 0:
        if joint_id in self.upper_body_ids:
          upper_joints.append(joints[joint_id])
        else:
          lower_joints.append(joints[joint_id])

    if np.random.randn() < 0.5 and len(upper_joints) > 2:
      selected_joints = upper_joints
    else:
      selected_joints = lower_joints \
          if len(lower_joints) > 2 else upper_joints

    if len(selected_joints) < 2:
      return None, None

    selected_joints = np.array(selected_joints, dtype=np.float32)
    center = selected_joints.mean(axis=0)[:2]

    left_top = np.amin(selected_joints, axis=0)
    right_bottom = np.amax(selected_joints, axis=0)

    w = right_bottom[0] - left_top[0] + 1
    h = right_bottom[1] - left_top[1] + 1

    if w > self.aspect_ratio * h:
      h = w * 1.0 / self.aspect_ratio
    elif w < self.aspect_ratio * h:
      w = h * self.aspect_ratio

    scale = np.array(
        [
            w * 1.0 / self.pixel_std,
            h * 1.0 / self.pixel_std
        ],
        dtype=np.float32
    )

    scale = scale * 1.5

    return center, scale

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.image_set_index))
    if self.shuffle == True:
      np.random.shuffle(self.image_set_index)

  def __len__(self,):
    return int(np.floor(len(self.image_set_index)/self.batch_size))

  def __getitem__(self, idx):
    # Generate indexes of the batch
    indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
    # Find list of IDs
    X, target,target_weight, meta = self.data_generation(indexes)
    return np.array(X,'float32'),np.array(target,'float32'), \
            np.array(target_weight,'float32'),meta




  def data_generation(self, list_IDs_temp):
    # Initialization
    batch_images = np.zeros(shape=(self.batch_size, self.image_size[1], self.image_size[0], 3), dtype=np.float)
    batch_heatmaps = np.zeros(shape=(self.batch_size,self.output_size[1], self.output_size[0], 17), dtype=np.float)
    batch_weights= np.zeros(shape=(self.batch_size,17,1), dtype=np.float)
    batch_metainfo = []
    count = 0

    for i in list_IDs_temp:
      X_temp,target_temp,target_weight_temp, meta_temp=self.load_single_batch(i)
      
      if X_temp is None:
        continue

      index = count % self.batch_size
      batch_images[index, :, :, :] = np.array(X_temp,'float32')
      batch_heatmaps[index, :, :, :] = np.array(target_temp,'float32')
      batch_weights[index, :, :] = np.array(target_weight_temp,'float32')
      batch_metainfo.append(meta_temp)
      count = count + 1

    return batch_images, batch_heatmaps, batch_weights,batch_metainfo

  def load_single_batch(self,id):

    db_rec = copy.deepcopy(self.db[id])

    image_file = db_rec['image']
    filename = db_rec['filename'] if 'filename' in db_rec else ''
    imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

    if self.data_format == 'zip':
      from utils import zipreader
      data_numpy = zipreader.imread(
          image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    else:
      data_numpy = cv2.imread(
          image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
      )
    if self.color_rgb:
      data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

    if data_numpy is None:
      logger.error('=> fail to read {}'.format(image_file))
      raise ValueError('Fail to read {}'.format(image_file))

    joints = db_rec['joints_3d']
    joints_vis = db_rec['joints_3d_vis']

    c = db_rec['center']
    s = db_rec['scale']
    score = db_rec['score'] if 'score' in db_rec else 1
    r = 0

    if self.is_train:
      if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
          and np.random.rand() < self.prob_half_body):
        c_half_body, s_half_body = self.half_body_transform(
            joints, joints_vis)

        if c_half_body is not None and s_half_body is not None:
          c, s = c_half_body, s_half_body

      sf = self.scale_factor
      rf = self.rotation_factor
      s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
        if random.random() <= 0.6 else 0

      if self.flip and random.random() <= 0.5:
        data_numpy = data_numpy[:, ::-1, :]
        joints, joints_vis = fliplr_joints(
            joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
        c[0] = data_numpy.shape[1] - c[0] - 1

    joints_heatmap = joints.copy()
    trans = get_affine_transform(c, s, r, self.image_size)
    trans_heatmap = get_affine_transform(c, s, r, self.heatmap_size)

    # Initialization
    input = np.asarray(cv2.warpAffine(
        data_numpy,
        trans,
        (int(self.image_size[0]), int(self.image_size[1])),
        flags=cv2.INTER_LINEAR),'float32')
 
    if self.transform:
      input = self.transform(input)

    for i in range(self.num_joints):
      if joints_vis[i, 0] > 0.0:
        joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        joints_heatmap[i, 0:2] = affine_transform(joints_heatmap[i, 0:2], trans_heatmap)

    target, target_weight = self.generate_target(joints_heatmap, joints_vis)

    meta = {
        'image': image_file,
        'filename': filename,
        'imgnum': imgnum,
        'joints': joints,
        'joints_vis': joints_vis,
        'center': c,
        'scale': s,
        'rotation': r,
        'score': score
    }

    target=np.moveaxis(target, 0, 2)
    target = np.asarray(target, 'float32')
    target_weight = np.asarray(target_weight, 'float32')

    return input, target, target_weight, meta



  def select_data(self, db):
    db_selected = []
    for rec in db:
      num_vis = 0
      joints_x = 0.0
      joints_y = 0.0
      for joint, joint_vis in zip(
              rec['joints_3d'], rec['joints_3d_vis']):
        if joint_vis[0] <= 0:
          continue
        num_vis += 1

        joints_x += joint[0]
        joints_y += joint[1]
      if num_vis == 0:
        continue

      joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

      area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
      joints_center = np.array([joints_x, joints_y])
      bbox_center = np.array(rec['center'])
      diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
      ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

      metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
      if ks > metric:
        db_selected.append(rec)

    logger.info('=> num db: {}'.format(len(db)))
    logger.info('=> num selected db: {}'.format(len(db_selected)))
    return db_selected


  def generate_target(self, joints, joints_vis):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''

    target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    assert self.target_type == 'gaussian', \
        'Only support gaussian map now!'

    if self.target_type == 'gaussian':
      target = np.zeros((self.num_joints,
                          self.heatmap_size[1],
                          self.heatmap_size[0]),
                        dtype=np.float32)

      tmp_size = self.sigma * 3

      for joint_id in range(self.num_joints):
        target_weight[joint_id] = \
        self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)

        if target_weight[joint_id] == 0:
          continue

        mu_x = joints[joint_id][0]
        mu_y = joints[joint_id][1]

        x = np.arange(0, self.heatmap_size[0], 1, np.float32)
        y = np.arange(0, self.heatmap_size[1], 1, np.float32)
        y = y[:, np.newaxis]

        v = target_weight[joint_id]
        if v > 0.5:
          target[joint_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))

    if self.use_different_joints_weight:
      target_weight = np.multiply(target_weight, self.joints_weight)

    return target, target_weight


  def adjust_target_weight(self, joint, target_weight, tmp_size):
    # feat_stride = self.image_size / self.heatmap_size
    mu_x = joint[0]
    mu_y = joint[1]
    # Check that any part of the gaussian is in-bounds
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
          or br[0] < 0 or br[1] < 0:
      # If not, just return the image as is
      target_weight = 0

    return target_weight
