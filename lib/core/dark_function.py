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

def train(config, train_loader, model, criterion, optimizer,epoch,
          output_dir='', tb_log_dir='', writer_dict=''):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()



    model.summary()
    end = time.time()

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
    image_path = []
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

            input_flipped = tf.reverse(input, 3).copy()
            input_flipped = input_flipped.numpy()
            outputs_flipped = model.evaluate(input_flipped)

            if isinstance(outputs_flipped, list):
                output_flipped = outputs_flipped[-1]
            else:
                output_flipped = outputs_flipped

            output_flipped = tf.reverse(output_flipped,
                                       val_dataset.flip_pairs)
            output_flipped = output_flipped.numpy()

            output = (output + output_flipped) * 0.5

        loss = criterion(output, target, target_weight)

        loss=loss.numpy()

        num_images = input.size
        # measure accuracy and record loss
        losses.update(loss.item(), num_images)
        _, avg_acc, cnt, pred = accuracy(output,target)

        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        c=[]
        s=[]
        score=[]

        for batch_id in range(input.shape[0]):
            c.append(meta[batch_id]['center'])
            s.append(meta[batch_id]['scale'])
            score.append(meta[batch_id]['score'])
            image_path.extend(meta[batch_id]['image'])

        c=np.array(c)
        s=np.array(s)


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

            prefix = '{}_{}'.format(
                os.path.join(output_dir, 'val'), i
            )
            # save_debug_images(config, input, meta, target, pred*4, output,
            #                   prefix)


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
