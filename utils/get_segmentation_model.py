# from swinT import get_swin_B
# from pytorch_pretrained_vit import ViT, load_pretrained_weights
import torchvision
import torch
import os
import timm
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append('segment-anything')
from segment_anything import sam_model_registry


def get_model(model, model_type, model_config=None):
    if 'hq' not in model.lower():
        sys.path.append('sam2')
    else: 
        sys.path.append('sam-hq/sam-hq2')
    
    from sam2.build_sam import build_sam2    
    
    if model.lower() == 'sam':
        if model_type == "vit_b": sam_checkpoint = "pretrained/sam_vit_b_01ec64.pth"
        if model_type == "vit_l": sam_checkpoint = "pretrained/sam_vit_l_0b3195.pth"
        if model_type == "vit_h": sam_checkpoint = "pretrained/sam_vit_h_4b8939.pth"
        device = "cuda"
        print("sb", model_type, sam_checkpoint)
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        return sam
    
    elif model.lower() == 'sam2':

        if model_type == 'vit_t': sam_checkpoint = "pretrained/sam2_hiera_tiny.pt"
        device = "cuda"
        sam = build_sam2(model_config, sam_checkpoint)
        sam.to(device=device)
        return sam

    elif model.lower() == 'sam2.1':
        if model_type == 'vit_t': sam_checkpoint = "pretrained/sam2.1_hiera_tiny.pt"
        device = "cuda"
        sam = build_sam2(model_config, sam_checkpoint)
        sam.to(device=device)
        return sam
    
    elif model.lower() == 'hq-sam2.1':
        if model_type == 'vit_l': sam_checkpoint = "pretrained/sam2.1_hq_hiera_large.pt"
        device = "cuda"
        sam = build_sam2(model_config, sam_checkpoint)
        
        sam.to(device=device)
        return sam
    
    elif model.lower() == 'asam2':
        if model_type == 'vit_t': sam_checkpoint = "pretrained/sam2_hiera_tiny.pt"
        sam_checkpoint = "pretrained/asam++.pth"
        device = "cuda"
        sam = build_sam2(model_config, sam_checkpoint)
        sam.to(device=device)
        return sam
    
    elif model.lower() == 'sam_efficient':
        if model_type == "vit_t": sam_checkpoint = torch.load("pretrained/efficient_sam_vitt.pt")['model']
        if model_type == "vit_s": sam_checkpoint = torch.load("pretrained/efficient_sam_vits.pt")['model']
        
        if model_type == 'vit_t': sam = build_efficient_sam_vitt()
        if model_type == 'vit_s': sam = build_efficient_sam_vits()      
        sam.load_state_dict(sam_checkpoint)
        return sam
