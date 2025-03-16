import matplotlib.pyplot as plt
import numpy as np
import torch

# Display dual time-phase image comparison
def show_image_pair(image_A: torch.Tensor, image_B: torch.Tensor, title: str = "Image Pair"):  
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  
    axes[0].imshow(image_A.permute(1, 2, 0).numpy())  
    axes[0].set_title("Time T1")  
    axes[1].imshow(image_B.permute(1, 2, 0).numpy())  
    axes[1].set_title("Time T2")  
    plt.suptitle(title)  
    plt.show()  

# Overlay change mask to original image
def overlay_change_mask(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.5):  
    image_np = image.permute(1, 2, 0).numpy()  
    mask_np = mask.squeeze().numpy()  
    plt.imshow(image_np)  
    plt.imshow(mask_np, alpha=alpha, cmap="Reds")  
    plt.title("Change Mask Overlay")  
    plt.axis("off")  
    plt.show()  

# Comparing Predictive Masks to Real Labels 
def plot_comparison(pred_mask: torch.Tensor, gt_mask: torch.Tensor, title: str = "Change Detection Result"):
    # 确保输入为二维张量 (H, W)
    pred_mask_np = pred_mask.detach().cpu().numpy().squeeze()
    gt_mask_np = gt_mask.detach().cpu().numpy().squeeze()
    
    # 直接截断到 [0,1] 范围（如果锐化导致超出范围）
    pred_mask_np = np.clip(pred_mask_np, 0, 1)
    gt_mask_np = np.clip(gt_mask_np, 0, 1)
    
    # 绘制图像
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pred_mask_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Predicted Mask")
    axes[1].imshow(gt_mask_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    plt.suptitle(title)
    plt.tight_layout()
    return fig