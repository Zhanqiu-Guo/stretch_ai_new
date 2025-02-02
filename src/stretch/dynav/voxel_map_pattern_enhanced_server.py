# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import datetime
import os
import pickle
import threading

# from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import scipy
import torch

# from stretch.utils.morphology import get_edges
import torch.nn.functional as F
from matplotlib import pyplot as plt

import stretch.utils.logger as logger
from stretch.agent.zmq_client import HomeRobotZmqClient as RobotClient
from stretch.core import get_parameters
from stretch.dynav.communication_util import load_socket, recv_array, recv_everything, send_array
from stretch.dynav.voxel_map_pattern_enhanced_localizer import RobustDynaMemLocalizer
from stretch.dynav.voxel_map_server import ImageProcessor
from stretch.mapping.voxel import SparseVoxelMapDynamem as SparseVoxelMap
from stretch.mapping.voxel import (
    SparseVoxelMapNavigationSpaceDynamem as SparseVoxelMapNavigationSpace,
)
from stretch.motion.algo.a_star import AStar
from stretch.perception.encoders import CustomImageTextEncoder, MaskSiglipEncoder
import time

class EnhancedImageProcessor(ImageProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace standard localizer with robust version
        self.voxel_map_localizer = RobustDynaMemLocalizer(
            self.voxel_map,
            clip_model=self.encoder.model,
            processor=self.encoder.processor,
            device=self.device,
            siglip=True
        )

    # def process_rgbd_images(self, rgb, depth, intrinsics, pose):
    #     with self.voxel_map_lock:
    #         if self.voxel_map_localizer.voxel_pcd._points is not None:
    #             old_instances = {k: v for k, v in self.voxel_map_localizer.object_instances.items()}
    #             self.voxel_map_localizer._update_instances(old_instances)
    #         else:
    #             old_instances = {}
                
    #     super().process_rgbd_images(rgb, depth, intrinsics, pose)

    #     # print("Update obstacle map as usual")
    #     # self.voxel_map.add(
    #     #     camera_pose=torch.Tensor(pose),
    #     #     rgb=torch.Tensor(rgb).permute(1, 2, 0),
    #     #     depth=torch.Tensor(depth),
    #     #     camera_K=torch.Tensor(intrinsics),
    #     # )

    #     if self.rerun and self.rerun_visualizer:
    #         self._update_visualization()

    def process_rgbd_images(self, rgb, depth, intrinsics, pose):
        if self.voxel_map_localizer.voxel_pcd._points is not None:
            old_points = self.voxel_map_localizer.voxel_pcd._points
            old_features = self.voxel_map_localizer.voxel_pcd._features
        else:
            old_points = None

        super().process_rgbd_images(rgb, depth, intrinsics, pose)

        if old_points is not None and self.voxel_map_localizer.voxel_pcd._points is not None:
            new_points = self.voxel_map_localizer.voxel_pcd._points
            new_features = self.voxel_map_localizer.voxel_pcd._features
            
            # Compute pairwise distances between old and new points
            dists = torch.cdist(old_points, new_points)
            min_dists, nn_indices = dists.min(dim=1)

            # Only process points that moved significantly
            moved_mask = min_dists > self.voxel_map_localizer.movement_threshold
            moved_indices = torch.where(moved_mask)[0]
            if len(moved_indices) == 0:
                return
            feat_keys = self._quantize_features_batch(old_features[moved_indices])
            new_positions = new_points[nn_indices[moved_indices]]
            current_time = time.time()
            self.voxel_map_localizer.feature_movements.update({
                tuple(key): (pos, current_time)
                for key, pos in zip(feat_keys, new_positions)
            })

    def _update_visualization(self):
        if self.voxel_map.voxel_pcd._points is not None:
            self.rerun_visualizer.update_voxel_map(space=self.space)
        for instance in self.voxel_map_localizer.object_instances.values():
            self.rerun_visualizer.log_custom_pointcloud(
                "world/tracked_objects",
                instance.center.cpu(),
                torch.Tensor([0, 1, 0]),  # Green for current position
                0.05
            )

            # Visualize recent movements
            if instance.movement_history:
                recent_moves = [m for m in instance.movement_history 
                              if time.time() - m[2] < 3600]
                for from_pos, to_pos, _ in recent_moves:
                    self.rerun_visualizer.log_arrow3D(
                        "world/object_movements",
                        [from_pos.cpu()],
                        [(to_pos - from_pos).cpu()],
                        torch.Tensor([1, 0, 0]),
                        0.02
                    )

    def sample_navigation(self, start, point, mode="navigation"):
        """Enhanced navigation considering object movements."""
        goal = super().sample_navigation(start, point, mode)

        if mode == "navigation" and point is not None:
            # Check if point corresponds to tracked object
            best_instance = None
            min_dist = float('inf')
            for instance in self.voxel_map_localizer.object_instances.values():
                dist = torch.norm(point - instance.center)
                if dist < min_dist and dist < 0.5:  # Within 50cm
                    min_dist = dist
                    best_instance = instance

            if best_instance and best_instance.movement_history:
                # Check recent movements
                recent_moves = [m for m in best_instance.movement_history 
                              if time.time() - m[2] < 300]  # Past 5 minutes
                if recent_moves:
                    # Consider navigating to recent movement destination
                    new_point = recent_moves[-1][1]
                    alt_goal = self.space.sample_target_point(
                        start, new_point, self.planner
                    )
                    if alt_goal is not None:
                        return alt_goal

        return goal
def main():
    torch.manual_seed(1)
    imageProcessor = EnhancedImageProcessor(
        log="dynamem_log/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    try:
        while True:
            imageProcessor.recv_text()
    except KeyboardInterrupt:
        imageProcessor.write_to_pickle()