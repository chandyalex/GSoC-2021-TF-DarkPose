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

import time
import logging
import os

import numpy as np
import tensorflow as tf

from core.evaluation import accuracy
from core.inference_dark import get_final_preds
from utils.transforms import flip_back



logger = logging.getLogger(__name__)

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir='', tb_log_dir='', writer_dict=''):

  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  acc = AverageMeter()
  
  model.summary()
  end = time.time()
  start_time = epoch_start_time = time.time()
  steps_per_epoch = 20

  for step, (input, target, target_weight, meta) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:

      # Run the forward pass of the layer.
      # The operations that the layer applies
      # to its inputs are going to be recorded
      # on the GradientTape.
      outputs = model(input, training=True)  # Logits for this minibatch
      # Compute the loss value for this minibatch.
      if isinstance(outputs, list):
        loss = criterion(outputs[0], target, target_weight)
        for output in outputs[1:]:
          loss += criterion(output, target, target_weight)
      else:
        output = outputs
        loss = criterion(output, target, target_weight)
  # Use the gradient tape to automatically retrieve
  # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss, model.trainable_weights)
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.distributed_apply(zip(grads, model.trainable_weights))
    #optimizer.apply_gradients(zip(grads, model.trainable_weights))

    loss = loss.numpy()


    losses.update(loss.item(), input.shape[0])

    _, avg_acc, cnt, pred = accuracy(output, target)
    acc.update(avg_acc, cnt)

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % config.PRINT_FREQ == 0:
      msg = 'Epoch: [{0}][{1}/{2}]\t' \
            'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
            'Speed {speed:.1f} samples/s\t' \
            'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
            'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
            'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, step, len(train_loader), batch_time=batch_time,
                speed=input.shape[0]/batch_time.val,
                data_time=data_time, loss=losses, acc=acc)
      print(msg)

def validate(config, val_loader, model, criterion, output_dir='',
             tb_log_dir='', writer_dict=None):

  batch_time = AverageMeter()
  losses = AverageMeter()
  acc = AverageMeter()

  num_samples = config.TRAIN.BATCH_SIZE_PER_GPU
  all_preds = np.zeros(
      (num_samples, config.MODEL.NUM_JOINTS, 3),
      dtype=np.float32
  )
  all_boxes = np.zeros((num_samples, 6))
  
  filenames = []
  imgnums = []
  idx = 0
  end = time.time()
  for i, (input, target, target_weight, meta) in enumerate(val_loader):
    outputs = model(input, training=False)
    output = outputs.numpy()
    if isinstance(outputs, list):
      output = outputs[-1]
    else:
      output = outputs
    output = output.numpy()


    if config.TEST.FLIP_TEST:
    
      input_flipped = np.flip(input, 3).copy()
    
  

      outputs_flipped = model(input_flipped, training=False)

      if isinstance(outputs_flipped, list):
        output_flipped = outputs_flipped[-1]
      else:
        output_flipped = outputs_flipped
      
      output_flipped = output_flipped.numpy()

      output_flipped = flip_back(output_flipped, val_loader.flip_pairs)
      

      output = (output + output_flipped) * 0.5

      loss = criterion(output, target, target_weight)

      loss = loss.numpy()

      num_images = input.shape[0]
      # measure accuracy and record loss
      losses.update(loss.item(), num_images)
      _, avg_acc, cnt, pred = accuracy(output, target)

      acc.update(avg_acc, cnt)

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      c = []
      s = []
      score = []
      meta_ = []
      image_path = []
   

      for batch_id in range(input.shape[0]):
        c.append(meta[batch_id]['center'])
        s.append(meta[batch_id]['scale'])
        score.append(meta[batch_id]['score'])
        image_path.append(meta[batch_id]['image'])

  

      c = np.array(c)
      s = np.array(s)

 

      preds, maxvals = get_final_preds(config, output, c, s)

      all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
      all_preds[idx:idx + num_images, :, 2:3] = maxvals
      # double check this all_boxes parts
      all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
      all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
      all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
      all_boxes[idx:idx + num_images, 5] = score
      

      # idx = idx+num_images

      if i % config.PRINT_FREQ == 0:
        msg = 'Test: [{0}/{1}]\t' \
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
              'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                  i, len(val_loader), batch_time=batch_time,
                  loss=losses, acc=acc)
        logger.info(msg)
        print(msg)

        # prefix = '{}_{}'.format(
        #     os.path.join(output_dir, 'val'), i

        # )
  name_values, perf_indicator = val_loader.evaluate(
      config, all_preds, output_dir, all_boxes, image_path,
      filenames, imgnums
  )
  return perf_indicator
      

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count if self.count != 0 else 0
