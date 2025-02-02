# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Optional

from typing import List, Optional, Tuple
import clip
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from torch import Tensor

# from ultralytics import YOLOWorld
# from transformers import AutoModel, AutoProcessor, CLIPTokenizer, Owlv2ForObjectDetection
from transformers import AutoModel, AutoProcessor

from stretch.perception.detection.owl import OwlPerception
from stretch.utils.logger import Logger
from stretch.utils.voxel import VoxelizedPointcloud
from stretch.dynav.voxel_map_localizer import VoxelMapLocalizer

from segment_anything import sam_model_registry, SamPredictor
import time

# Create a logger
logger = Logger(__name__)


def get_inv_intrinsics(intrinsics):
    # return intrinsics.double().inverse().to(intrinsics)
    fx, fy, ppx, ppy = (
        intrinsics[..., 0, 0],
        intrinsics[..., 1, 1],
        intrinsics[..., 0, 2],
        intrinsics[..., 1, 2],
    )
    inv_intrinsics = torch.zeros_like(intrinsics)
    inv_intrinsics[..., 0, 0] = 1.0 / fx
    inv_intrinsics[..., 1, 1] = 1.0 / fy
    inv_intrinsics[..., 0, 2] = -ppx / fx
    inv_intrinsics[..., 1, 2] = -ppy / fy
    inv_intrinsics[..., 2, 2] = 1.0
    return inv_intrinsics


def get_xyz(depth, pose, intrinsics):
    """Returns the XYZ coordinates for a set of points.

    Args:
        depth: The depth array, with shape (B, 1, H, W)
        pose: The pose array, with shape (B, 4, 4)
        intrinsics: The intrinsics array, with shape (B, 3, 3)

    Returns:
        The XYZ coordinates of the projected points, with shape (B, H, W, 3)
    """
    if not isinstance(depth, torch.Tensor):
        depth = torch.from_numpy(depth)
    if not isinstance(pose, torch.Tensor):
        pose = torch.from_numpy(pose)
    if not isinstance(intrinsics, torch.Tensor):
        intrinsics = torch.from_numpy(intrinsics)
    while depth.ndim < 4:
        depth = depth.unsqueeze(0)
    while pose.ndim < 3:
        pose = pose.unsqueeze(0)
    while intrinsics.ndim < 3:
        intrinsics = intrinsics.unsqueeze(0)
    (bsz, _, height, width), device, dtype = depth.shape, depth.device, intrinsics.dtype

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(0, width, device=device, dtype=dtype),
        torch.arange(0, height, device=device, dtype=dtype),
        indexing="xy",
    )
    xy = torch.stack([xs, ys], dim=-1).flatten(0, 1).unsqueeze(0).repeat_interleave(bsz, 0)
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)

    # Applies intrinsics and extrinsics.
    # xyz = xyz @ intrinsics.inverse().transpose(-1, -2)
    xyz = xyz @ get_inv_intrinsics(intrinsics).transpose(-1, -2)
    xyz = xyz * depth.flatten(1).unsqueeze(-1)
    xyz = (xyz[..., None, :] * pose[..., None, :3, :3]).sum(dim=-1) + pose[..., None, :3, 3]

    xyz = xyz.unflatten(1, (height, width))

    return xyz

class ObjectInstanceMemory:
    
    def __init__(self, decay_factor=0.95, confidence_threshold=0.5):
        self.instances = {} 
        self.location_memory = {} 
        self.decay_factor = decay_factor
        self.confidence_threshold = confidence_threshold
        
    def update_instance(self, category: str, instance_id: int, location: np.ndarray, 
                       confidence: float, timestamp: float):
        key = f"{category}_{instance_id}"
        
        if key not in self.instances:
            self.instances[key] = {
                'locations': [],
                'confidences': [],
                'timestamps': [],
                'last_seen': None
            }
            
        instance = self.instances[key]
        instance['locations'].append(location)
        instance['confidences'].append(confidence)
        instance['timestamps'].append(timestamp)
        instance['last_seen'] = location

        if category not in self.location_memory:
            self.location_memory[category] = {
                'locations': [],
                'visit_counts': [],
                'last_updates': []
            }
            
        self._update_location_memory(category, location, timestamp)
        
    def _update_location_memory(self, category: str, location: np.ndarray, timestamp: float):
        memory = self.location_memory[category]

        merged = False
        for i, known_loc in enumerate(memory['locations']):
            if np.linalg.norm(location - known_loc) < 0.5:  # 0.5m threshold
                memory['locations'][i] = (memory['locations'][i] * memory['visit_counts'][i] + location) / (memory['visit_counts'][i] + 1)
                memory['visit_counts'][i] += 1
                memory['last_updates'][i] = timestamp
                merged = True
                break
                
        if not merged:
            memory['locations'].append(location)
            memory['visit_counts'].append(1)
            memory['last_updates'].append(timestamp)
            
    def get_search_locations(self, category: str, current_time: float) -> List[np.ndarray]:
        
        if category not in self.location_memory:
            return []
            
        memory = self.location_memory[category]
        scores = []
        
        for i, location in enumerate(memory['locations']):
            # Calculate score based on visit count and time decay
            time_factor = np.exp(-0.1 * (current_time - memory['last_updates'][i]))
            score = memory['visit_counts'][i] * time_factor
            scores.append((score, location))
            
        scores.sort(reverse=True)
        return [loc for _, loc in scores]

class EnhancedVoxelMapLocalizer(VoxelMapLocalizer):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instance_memory = ObjectInstanceMemory()
        self.sam_model = sam_model_registry["vit_h"](checkpoint="/data/zhanqiu/sam_vit_h_4b8939.pth")
        self.sam_model.to(device=self.device)
        self.predictor = SamPredictor(self.sam_model)
        
    def process_image_with_sam(self, rgb: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        self.predictor.set_image(rgb)

        masks, scores, _ = self.predictor.predict()
        return masks, scores
        
    def update_memory(self, category: str, rgb: np.ndarray, depth: np.ndarray, 
                     camera_pose: np.ndarray, camera_K: np.ndarray):
        masks, scores = self.process_image_with_sam(rgb)
        current_time = time.time()
        
        for mask, score in zip(masks, scores):
            if score < self.instance_memory.confidence_threshold:
                continue
                
            instance_depth = depth[mask]
            if len(instance_depth) == 0:
                continue
                
            centroid_3d = self._compute_3d_centroid(mask, instance_depth, camera_pose, camera_K)
            instance_id = self._assign_instance_id(centroid_3d, category)
            
            self.instance_memory.update_instance(
                category=category,
                instance_id=instance_id,
                location=centroid_3d,
                confidence=score,
                timestamp=current_time
            )
            
    def _compute_3d_centroid(self, mask: np.ndarray, depth: np.ndarray, 
                            camera_pose: np.ndarray, camera_K: np.ndarray) -> np.ndarray:
        xyz = get_xyz(depth[mask], pose, intrinsics)
        B, H, W, _ = xyz.shape
        points_reshaped = points.reshape(B, H*W, 3)
        centroids = np.mean(points_reshaped, axis=1)

        return centroids
        
    def _assign_instance_id(self, location: np.ndarray, category: str) -> int:
        DISTANCE_THRESHOLD = 0.5
        TIME_THRESHOLD = 300  # 5 minutes
        
        current_time = time.time()
        
        category_instances = {
            instance_id: data 
            for instance_id, data in self.instance_memory.instances.items() 
            if instance_id.startswith(f"{category}_")
        }
        
        if not category_instances:
            return self._create_new_instance_id(category)

        distances = []
        instance_ids = []
        
        for instance_id, data in category_instances.items():
            if data['last_seen'] is not None and data['timestamps']:
                time_since_last_seen = current_time - data['timestamps'][-1]
                
                if time_since_last_seen > TIME_THRESHOLD:
                    continue
                    
                dist = np.linalg.norm(location - data['last_seen'])
                distances.append(dist)
                instance_ids.append(int(instance_id.split('_')[1]))
        
        if not distances:
            return self._create_new_instance_id(category)

        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        if min_dist <= DISTANCE_THRESHOLD:
            return instance_ids[min_dist_idx]
        else:
            return self._create_new_instance_id(category)

    def _create_new_instance_id(self, category: str) -> int:
        existing_ids = set()
        prefix = f"{category}_"
        
        for instance_id in self.instance_memory.instances.keys():
            if instance_id.startswith(prefix):
                id_num = int(instance_id.split('_')[1])
                existing_ids.add(id_num)
        
        new_id = 1
        while new_id in existing_ids:
            new_id += 1
            
        return new_id
        
    def find_object(self, category: str) -> Optional[np.ndarray]:
        search_locations = self.instance_memory.get_search_locations(category, time.time())
        if not search_locations:
            return None
            
        return search_locations[0]