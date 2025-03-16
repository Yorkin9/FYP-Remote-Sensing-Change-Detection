#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utilities.data_loader import ChangeDetectionDataset
from models.change_detection import ChangeDetectionModel
from models.modeling.common import compute_metrics
from utilities.visualization import plot_comparison



def main():
    """
    Main testing script. We also reduce batch size here to avoid OOM
    if the resolution is large.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Path config
    TEST_DATA_DIR = os.path.join(project_root, "data/LEVIR_CD/test")
    CHECKPOINT_PATH = os.path.join(project_root, "results/checkpoints/best_model.pth")
    SAVE_PRED_DIR = os.path.join(project_root, "results/predictions")
    SAVE_VIS_DIR = os.path.join(project_root, "results/visualization")
    os.makedirs(SAVE_PRED_DIR, exist_ok=True)
    os.makedirs(SAVE_VIS_DIR, exist_ok=True)

    # Create dataset and dataloader
    test_dataset = ChangeDetectionDataset(
        root_dir=TEST_DATA_DIR,
        use_augmentation=False
    )
    # For inference, we can set batch_size=1 or 2 to reduce memory usage
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model (same config as training)
    model = ChangeDetectionModel(
        sam_type="vit_b",
        checkpoint=None,  # We'll load the fine-tuned weights below
        freeze_encoder=False
    ).to(device)

    # Load fine-tuned weights
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print("Loaded model weights from best_model.pth.")

    total_f1 = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            image_A, image_B, label, filenames = batch
            image_A = image_A.to(device)
            image_B = image_B.to(device)
            label = label.to(device)

            # (1) Forward pass to get raw logits
            logits = model(image_A, image_B)

            # (2) Convert to probabilities with sigmoid
            prob = torch.sigmoid(logits)

            # (3) Binarize at 0.5 threshold
            output_bin = (prob > 0.5).float()

            # (4) Visualization and saving predictions
            for i in range(output_bin.shape[0]):
                pred_mask = output_bin[i].cpu().squeeze()
                gt_mask = label[i].cpu().squeeze()

                torch.save(pred_mask, os.path.join(SAVE_PRED_DIR, f"{filenames[i]}_pred.pt"))
                
                fig = plot_comparison(pred_mask, gt_mask, title=f"Prediction vs GT: {filenames[i]}")
                fig.savefig(os.path.join(SAVE_VIS_DIR, f"{filenames[i]}.png"))
                plt.close(fig)

            # (5) Compute F1/IoU metrics
            metrics = compute_metrics(output_bin.squeeze(1), label.squeeze(1))
            total_f1 += metrics['f1']
            total_iou += metrics['iou']

    num_batches = len(test_loader)
    print("\nTest Results:")
    print(f"Average F1-score: {total_f1 / num_batches:.4f}")
    print(f"Average IoU:      {total_iou / num_batches:.4f}")

if __name__ == '__main__':
    main()
