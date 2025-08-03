import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import ASPP

class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone="resnet50", num_classes=9, output_stride=16, pretrained_backbone=False):
        super(DeepLabV3Plus, self).__init__()
        
        # Backbone (ResNet50 or ResNet101)
        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained_backbone)
        elif backbone == "resnet101":
            resnet = models.resnet101(pretrained=pretrained_backbone)
        else:
            raise ValueError("Unsupported backbone: {}".format(backbone))
        
        # Remove fully connected layers
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # ASPP (Atrous Spatial Pyramid Pooling)
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise ValueError("Unsupported output stride: {}".format(output_stride))
        
        self.aspp = ASPP(2048, dilations)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # Initialize weights
        self._init_weights()

    def forward(self, x):
        # Backbone (ResNet)
        x_low = self.backbone[:5](x)  # Low-level features (before layer3)
        x = self.backbone[5:](x_low)   # High-level features (layer3 + layer4)
        
        # ASPP
        x = self.aspp(x)
        x = nn.functional.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        
        # Decoder (combine with low-level features)
        x_low = self.decoder[0](x_low)
        x_low = self.decoder[1](x_low)
        x_low = self.decoder[2](x_low)
        
        x = torch.cat([x, x_low], dim=1)
        x = self.decoder[3:](x)
        
        # Upsample to input resolution
        x = nn.functional.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)