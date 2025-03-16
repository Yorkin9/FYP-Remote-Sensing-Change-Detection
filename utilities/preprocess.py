#!/usr/bin/env python
# coding: utf-8

import torchvision.transforms as transforms

# Define the data preprocessing process
def get_preprocess():
    return transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust image size
        transforms.ToTensor(),          # Convert to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
    ])

