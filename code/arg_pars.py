#!/usr/bin/env python

"""Hyper-parameters and logging set up
opt: include all hyper-parameters
logger: unified logger for the project
"""

__all__ = ['opt']
__author__ = 'Yuerong Li'
__date__ = 'March 2023'

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--video-path', default='../data/breakfast/videos/',help='Specify the path to your video file.')
parser.add_argument('--frame-path', default='../data/breakfast/frames',help='Specify the path to your frame file.')
parser.add_argument('--object-path', default='../data/breakfast/object_info',help='Specify the path to your object file.')
parser.add_argument('--video-info-file', default='../data/breakfast/video_info.pkl',help='Specify the path to your video info file.')
parser.add_argument('--node-info-file', default='../data/breakfast/new_relation.json',help='Specify the path to your gt feature file.')
parser.add_argument('--img_dim', default=224, type=int)
parser.add_argument('--output-path', default='../output/',help='Specify the path for output file.')
parser.add_argument('--dataset', default='BF',help='BF for breakfast')
parser.add_argument('--feature_model', default='tf',help='tf for global perception model; tf_object for human-object interaction model; gat for object relationship model')
parser.add_argument('--pkl_folder_name', default='', type=str)
parser.add_argument('--seq_len', default=5, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=2, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=1, type=int)
parser.add_argument('--mode', default='val', type=str)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')

#training setting
parser.add_argument('--num_epoch', default=4, type=int, help='number of epochs')
parser.add_argument('--training_lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--weight_decay',type=float,default=1e-4)
parser.add_argument('--step_size',type=list,default=[90000])#?
parser.add_argument('--step_gamma',type=float,default=0.1)
parser.add_argument('--ckpt_interval',type=int,default=10000)
parser.add_argument('--info_interval',type=int,default=200)
parser.add_argument('--save_interval',type=int,default=2000)
parser.add_argument('--feature_dim',type=int,default=2048)

###bdy_dec
parser.add_argument('--combine', default=False,type=bool)
parser.add_argument('--order', default=15,type=int)
parser.add_argument('--theta_1', default=1,type=float)
parser.add_argument('--theta_2', default=1,type=float)
parser.add_argument('--theta_3', default=1,type=float)
parser.add_argument('--theta_n', default=10,type=float)

###evaluation
parser.add_argument('--gt-path', default='../data/breakfast/output/groundTruth',help='Specify the path to your gt labels.')

#Run
parser.add_argument('--gpu', default='6,7,8,9', type=str)
parser.add_argument('--num-gpus', default=4, type=int)
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_workers', default=8, type=int)

opt = parser.parse_args()

