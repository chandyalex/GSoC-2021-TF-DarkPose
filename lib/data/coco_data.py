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
import sys

import numpy as np
from pycocotools.coco import COCO
from data import coco
import os
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    
    def __init__(self, cfg, batch_size=32, dim=(256,256,3), n_channels=3,
        n_classes=10, shuffle=True):
        'Initialization'

        self.dim = dim
        self.batch_size = batch_size

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.config=cfg
        self.dta_path=cfg.DATASET.ROOT
        self.data_set=cfg.DATASET.TRAIN_SET
        self.coco = COCO(self._get_ann_file_keypoint())
        self.list_IDs = self.coco.getImgIds()

        self.data= coco.COCODataset(cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        images, target, target_weights,meta = self.__data_generation(list_IDs_temp)

        return images, target, target_weights,meta


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization



        # Generate data

        for i in enumerate(list_IDs_temp):
            images, target, target_weights,meta=self.data[list_IDs_temp]
            # Store sample

            # X[i,] =
            #
            # # Store class
            # y[i] = self.labels[ID]

        return images, target, target_weights,meta

    def _get_ann_file_keypoint(self):
        """ self.root / annotations / person_keypoints_train2017.json """
        prefix = 'person_keypoints' \
            if 'test' not in 'person_keypoints_train2017' else 'image_info'
        return os.path.join(
            self.dta_path,
            'annotations',
            prefix + '_train' + '.json'
        )
