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
# from utils.vis import save_debug_images


logger = logging.getLogger(__name__)

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir='', tb_log_dir='', writer_dict=''):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.compile(loss=criterion, optimizer=optimizer,run_eagerly=True)

    model.summary()
    end = time.time()

    for epoch in range(epoch):
        print("\nStart of epoch %d" % (epoch,))
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
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss=loss.numpy()

            losses.update(loss.item(), input.size)

            _, avg_acc, cnt, pred = accuracy(output,target)
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss)))
                print("Seen so far: %s samples" % ((step + 1) * 64))





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
