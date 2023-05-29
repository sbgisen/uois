#!/usr/bin/env pipenv-shebang
# -*- coding:utf-8 -*-

# Copyright (c) 2023 SoftBank Corp.
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
#

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import glob
import json
import sys
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
# My libraries. Ugly hack to import from sister directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import src.data_augmentation as data_augmentation
import src.evaluation as evaluation
import src.segmentation as segmentation
import src.util.flowlib as flowlib
import src.util.utilities as util_
import torch

from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import rospy
import rospkg
import message_filters
from uois.srv import Inference, InferenceResponse, InferenceRequest

class Segmentation:
    def __init__(self) -> None:
        dsn_config = {
            # Sizes
            'feature_dim' : 64, # 32 would be normal

            # Mean Shift parameters (for 3D voting)
            'max_GMS_iters' : 10,
            'epsilon' : 0.05, # Connected Components parameter
            'sigma' : 0.02, # Gaussian bandwidth parameter
            'num_seeds' : 200, # Used for MeanShift, but not BlurringMeanShift
            'subsample_factor' : 5,

            # Misc
            'min_pixels_thresh' : 500,
            'tau' : 15.,
        }
        rrn_config = {
            # Sizes
            'feature_dim' : 64, # 32 would be normal
            'img_H' : 224,
            'img_W' : 224,

            # architecture parameters
            'use_coordconv' : False,
        }
        uois3d_config = {
            # Padding for RGB Refinement Network
            'padding_percentage' : 0.25,

            # Open/Close Morphology for IMP (Initial Mask Processing) module
            'use_open_close_morphology' : True,
            'open_close_morphology_ksize' : 9,

            # Largest Connected Component for IMP module
            'use_largest_connected_component' : True,
        }
        checkpoint_dir = '/home/gisen/soar/src/uois/checkpoints/' # TODO: change this to directory of downloaded models
        dsn_filename = checkpoint_dir + 'DepthSeedingNetwork_3D_TOD_checkpoint.pth'
        rrn_filename = checkpoint_dir + 'RRN_OID_checkpoint.pth'
        uois3d_config['final_close_morphology'] = 'TableTop_v5' in rrn_filename
        self.uois_net_3d = segmentation.UOISNet3D(uois3d_config, 
                                            dsn_filename,
                                            dsn_config,
                                            rrn_filename,
                                            rrn_config
                                            )
        self.bridge = CvBridge()
        camera_info = rospy.wait_for_message('~camera_info', CameraInfo)
        self.k = np.array(camera_info.K).reshape(3, 3)
        # subs = []
        # subs.append(message_filters.Subscriber('~rgb', Image))
        # subs.append(message_filters.Subscriber('~depth', Image))
        rospy.Service('~inference', Inference, self.inference)

        self.pub = rospy.Publisher('~seg_mask', Image, queue_size=1)
        self.pub2 = rospy.Publisher('~segmap', Image, queue_size=1)

        # sync = message_filters.ApproximateTimeSynchronizer(subs, 100, 0.1)
        # sync.registerCallback(self.inference)

        rospy.loginfo('segmentation Inference Ready')

    # def inference(self, rgb_msg, depth_msg):
    def inference(self, req: InferenceRequest):

        try:
            depth = self.bridge.imgmsg_to_cv2(req.depth, "passthrough")
            depth = self.depth2pc(depth, self.k)
            color = self.bridge.imgmsg_to_cv2(req.rgb, desired_encoding="bgr8")
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            color = data_augmentation.standardize_image(color)
        except CvBridgeError as e:
            rospy.logwarn(e)
            return

        batch = {
            'rgb' : data_augmentation.array_to_tensor(np.expand_dims(color, axis=0)),
            'xyz' : data_augmentation.array_to_tensor(np.expand_dims(depth, axis=0)),

        }

        fg_masks, center_offsets, initial_masks, seg_masks = self.uois_net_3d.run_on_batch(batch)

        seg_masks = seg_masks.cpu().numpy()
        # fg_masks = fg_masks.cpu().numpy()
        # center_offsets = center_offsets.cpu().numpy().transpose(0,2,3,1)
        # initial_masks = initial_masks.cpu().numpy()

        num_objs = np.unique(seg_masks[0, ...]).max() + 1
        seg_mask_plot = util_.get_color_mask(seg_masks[0, ...], nc=num_objs)
        rospy.loginfo(f'num_objs: {num_objs}, seg_mask_plot.shape: {seg_mask_plot.shape}')

        self.pub.publish(self.bridge.cv2_to_imgmsg(seg_mask_plot, encoding="bgr8"))
        result = self.bridge.cv2_to_imgmsg(seg_masks[0].astype(np.uint16))
        self.pub2.publish(result)
        resp = InferenceResponse()
        resp.segmap = result
        return resp

    def depth2pc(self, depth, K):
        """
        Convert depth and intrinsics to point cloud and optionally point cloud color
        :param depth: hxw depth map in m
        :param K: 3x3 Camera Matrix with intrinsics
        :returns: (H x W x 3 point cloud)
        """
        pc = np.zeros((depth.shape[0], depth.shape[1], 3))
        pc[:,:,0] = (depth-K[0,2])*depth/K[0,0]
        pc[:,:,1] = (depth-K[1,2])*depth/K[1,1]
        pc[:,:,2] = depth
        pc[np.isnan(pc)] = 0
        return pc

if __name__ == '__main__':
    rospy.init_node('uois')
    app = Segmentation()
    rospy.spin()