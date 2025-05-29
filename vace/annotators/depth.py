# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import os
from einops import rearrange

from .utils import convert_to_numpy, resize_image, resize_image_ori
from .midas.api import MiDaSInference
from ..model_utils import ensure_annotator_models_downloaded

class DepthAnnotator:
    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        models_dir = ensure_annotator_models_downloaded()
        pretrained_model = os.path.join(models_dir, 'VACE-Annotators', 'depth', 'dpt_hybrid-midas-501f0c75.pt')
        self.model = MiDaSInference(model_type='dpt_hybrid', model_path=pretrained_model).to(self.device)
        self.model.eval()
        self.a = cfg.get('A', np.pi * 2.0)
        self.bg_th = cfg.get('BG_TH', 0.1)

    def __call__(self, image):
        image = convert_to_numpy(image)
        image = resize_image(image, 384)
        with torch.no_grad():
            depth = self.model(image)
        depth = resize_image_ori(depth, image.shape[:2])
        return depth


class DepthVideoAnnotator(DepthAnnotator):
    def forward(self, frames):
        ret_frames = []
        for frame in frames:
            anno_frame = super().forward(np.array(frame))
            ret_frames.append(anno_frame)
        return ret_frames


class DepthV2Annotator:
    def __init__(self, cfg, device=None):
        from .depth_anything_v2.dpt import DepthAnythingV2
        models_dir = ensure_annotator_models_downloaded()
        pretrained_model = os.path.join(models_dir, 'VACE-Annotators', 'depth', 'depth_anything_v2_vitl.pth')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]).to(self.device)
        self.model.load_state_dict(
            torch.load(
                pretrained_model,
                map_location=self.device
            )
        )
        self.model.eval()

    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def forward(self, image):
        image = convert_to_numpy(image)
        depth = self.model.infer_image(image)

        depth_pt = depth.copy()
        depth_pt -= np.min(depth_pt)
        depth_pt /= np.max(depth_pt)
        depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

        depth_image = depth_image[..., np.newaxis]
        depth_image = np.repeat(depth_image, 3, axis=2)
        return depth_image


class DepthV2VideoAnnotator(DepthV2Annotator):
    def forward(self, frames):
        ret_frames = []
        for frame in frames:
            anno_frame = super().forward(np.array(frame))
            ret_frames.append(anno_frame)
        return ret_frames
