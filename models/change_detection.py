import torch
import torch.nn as nn
import torch.nn.functional as F
from .modeling.image_encoder import ImageEncoderViT
from .build_sam import build_sam_vit_b, build_sam_vit_l, build_sam_vit_h

class ChangeDetectionModel(nn.Module):
    """
    使用 SAM 的图像编码器来做双时相变化检测的示例模型。
    1) 分别对两张输入图像编码 -> feat_A, feat_B
    2) 拼接并融合 -> fused
    3) 上采样解码 -> 最终输出 1 通道 raw logits (未做 Sigmoid)
    """
    def __init__(self, sam_type="vit_b", checkpoint=None, freeze_encoder=True):
        super().__init__()

        # 1. 根据指定类型构建 SAM
        if sam_type == "vit_b":
            self.sam = build_sam_vit_b(checkpoint=checkpoint)
        elif sam_type == "vit_l":
            self.sam = build_sam_vit_l(checkpoint=checkpoint)
        elif sam_type == "vit_h":
            self.sam = build_sam_vit_h(checkpoint=checkpoint)
        else:
            raise ValueError(f"Unknown sam_type: {sam_type}")

        # 2. 冻结或微调编码器
        if freeze_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False

        # SAM 的 image_encoder 通常会有一个 out_chans (默认为256)，
        # 也可查看 build_sam.py 中 _build_sam() 里 image_encoder.out_chans
        out_chans = getattr(self.sam.image_encoder, "out_chans", 256)

        # 3. 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * out_chans, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 4. 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, kernel_size=1)
            # 注意：不做 Sigmoid，让输出是 raw logits
        )

    def forward(self, image_A, image_B):
        """
        image_A, image_B: [B,3,H,W]  (已做预处理/归一化)
        return: [B,1,H,W] (raw logits)
        """
        # 1) 提取特征
        feat_A = self.sam.image_encoder(image_A)  # [B, out_chans, H', W']
        feat_B = self.sam.image_encoder(image_B)  # [B, out_chans, H', W']

        # 2) 融合
        fused = torch.cat([feat_A, feat_B], dim=1)  # [B, 2*out_chans, H', W']
        fused = self.fusion(fused)                 # [B, 256, H', W']

        # 3) 解码
        logits = self.decoder(fused)               # [B, 1, H, W]
        return logits