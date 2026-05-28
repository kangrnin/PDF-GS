import torch
import torch.nn as nn
import warnings
from torchvision.transforms import v2
from transformers import AutoModel

warnings.filterwarnings("ignore", message=".*xFormers.*")


# DINOv3 ViT-B/16: 1 CLS + 4 register tokens precede patch tokens in last_hidden_state.
_DINOV3_NUM_SPECIAL = 1 + 4


class DINOv3FeatureExtractor(nn.Module):
    def __init__(self, repo='facebook/dinov3-vitb16-pretrain-lvd1689m'):
        super(DINOv3FeatureExtractor, self).__init__()
        self.backbone = AutoModel.from_pretrained(repo).cuda().eval()

    def forward(self, image, feature_size=50):
        with torch.no_grad():
            patch = 16
            target_hw = feature_size * patch
            transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((target_hw, target_hw), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            img_tensor = transform(image).unsqueeze(0).cuda()
            out = self.backbone(img_tensor)
            patch_tokens = out.last_hidden_state[:, _DINOV3_NUM_SPECIAL:, :]
            B, _, C = patch_tokens.shape
            dino_features = patch_tokens.view(B, feature_size, feature_size, C).permute(0, 3, 1, 2)
        return dino_features.squeeze()
