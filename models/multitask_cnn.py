import torch
import torch.nn as nn
from torchvision import models

class AgeGenderModel(nn.Module):
    def __init__(self, pretrained=True):
        super(AgeGenderModel, self).__init__()
        
        self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        num_ftrs = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Identity()

        self.gender_head = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

        self.age_head = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        
        gender_logits = self.gender_head(features)
        age_pred = self.age_head(features)
        
        return gender_logits, age_pred

if __name__ == "__main__":
    model = AgeGenderModel()
    dummy_input = torch.randn(1, 3, 224, 224) 
    g_out, a_out = model(dummy_input)
    print(f"Output giới tính (logits): {g_out.shape}")
    print(f"Output tuổi: {a_out.shape}")              