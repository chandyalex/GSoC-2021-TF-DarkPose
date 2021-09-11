from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

from config import cfg
from config import update_config
from utils.utils import create_logger

import numpy as np
import tensorflow as tf

import models
from data import coco

from core.loss import JointsMSELoss, JointsOHKMMSELoss, dice_loss
from models import pose_resnet
from core.evaluation import accuracy
from utils.utils import get_optimizer



from core.dark_function import train, validate

def parse_args():
  parser = argparse.ArgumentParser(description='Train keypoints network')
  # general
  parser.add_argument('--cfg',
                      help='experiment configure file name',
                      required=True,
                      type=str)

  parser.add_argument('opts',
                      help="Modify config options using the command-line",
                      default=None,
                      nargs=argparse.REMAINDER)

  parser.add_argument('--modelDir',
                      help='model directory',
                      type=str,
                      default='')
  parser.add_argument('--logDir',
                      help='log directory',
                      type=str,
                      default='')
  parser.add_argument('--dataDir',
                      help='data directory',
                      type=str,
                      default='')
  parser.add_argument('--prevModelDir',
                      help='prev Model directory',
                      type=str,
                      default='')

  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  update_config(cfg, args)

  logger, final_output_dir, tb_log_dir = create_logger(
      cfg, args.cfg, 'train')

  logger.info(pprint.pformat(args))
  logger.info(cfg)


  # copy model file
  this_dir = os.path.dirname(__file__)
  shutil.copy2(
      os.path.join(this_dir, 'models', cfg.MODEL.NAME + '.py'),
      final_output_dir)

  train_dataset = coco.COCODataset(cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True)

  valid_dataset = coco.COCODataset(cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False)

  print(len(valid_dataset))
  print("validation datat set length")

  dump_input = np.random.rand(
                1, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0],3
                )
    


  model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
      cfg, is_train=True
  )

  model.build(input_shape=dump_input.shape)

  # define loss function (criterion) and optimizer
  optimizer = get_optimizer(cfg, model)
  criterion = JointsMSELoss

  model.compile(optimizer, criterion,run_eagerly=True)

  best_perf = 0.0
  best_model = False
  last_epoch = -1
  optimizer = get_optimizer(cfg, model)
  begin_epoch = cfg.TRAIN.BEGIN_EPOCH
  checkpoint_file = os.path.join(
      final_output_dir, 'checkpoint.h5'
  )

  if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
    logger.info("=> loading checkpoint '{}'".format(checkpoint_file))

  for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

    train(cfg, train_dataset, model, criterion, optimizer, 
            epoch, final_output_dir, tb_log_dir)
    # evaluate on validation set
    
    perf_indicator = validate(
        cfg, valid_dataset, model, criterion,
        final_output_dir, tb_log_dir)


    if perf_indicator >= best_perf:
      best_perf = perf_indicator
      best_model = True
    else:
      best_model = False

    logger.info('=> saving checkpoint to {}'.format(final_output_dir))


  final_model_state_file = os.path.join(final_output_dir, 'final_state')

  logger.info('=> saving final model state to {}'.format(
      final_model_state_file))

  tf.keras.models.save_model(model, final_model_state_file)



if __name__ == '__main__':
  main()