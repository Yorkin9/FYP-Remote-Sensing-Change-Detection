#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utilities.data_loader import ChangeDetectionDataset
from models.change_detection import ChangeDetectionModel
from models.modeling.common import HybridLoss



def main():
    """
    Main training script with smaller batch size and reduced image resolution
    to mitigate CUDA OOM issues.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Path configuration
    TRAIN_DATA_DIR = os.path.join(project_root, "data/LEVIR_CD/train")
    CHECKPOINT_DIR = os.path.join(project_root, "results/checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Create dataset and dataloader
    train_dataset = ChangeDetectionDataset(
        root_dir=TRAIN_DATA_DIR,
        use_augmentation=True  # random flips, rotations, etc.
    )
    # Decrease batch_size to avoid OOM
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    # Load the official SAM weights from models/weights
    SAM_CHECKPOINT = os.path.join(project_root, "models", "weights", "sam_vit_b.pth")
    model = ChangeDetectionModel(
        sam_type="vit_b",
        checkpoint=SAM_CHECKPOINT,
        freeze_encoder=False
    ).to(device)

    # Loss and optimizer
    # HybridLoss = BCEWithLogits + Dice
    criterion = HybridLoss(alpha=0.7, pos_weight=20.0).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Use GradScaler for mixed-precision
    scaler = GradScaler()

    # Training config
    NUM_EPOCHS = 30
    best_loss = float("inf")
    patience = 5
    no_improve = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc="Training", unit="batch"):
            image_A, image_B, label, _ = batch
            image_A = image_A.to(device, non_blocking=True)
            image_B = image_B.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad()
            # Mixed precision context
            with autocast():
                logits = model(image_A, image_B)  # raw logits
                loss = criterion(logits, label)

            # Backprop with GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"New best model saved with loss: {best_loss:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print("\nTraining completed!")

if __name__ == '__main__':
    main()
