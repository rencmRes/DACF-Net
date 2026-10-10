import torch.nn as nn
import torchvision

from . import block as B


class DVCFDiscriminator(nn.Module):
    """VGG-style discriminator for 256x256 one-channel training patches."""

    def __init__(self, in_nc=1, base_nf=64, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(DVCFDiscriminator, self).__init__()
        channels = [in_nc, base_nf, base_nf, base_nf * 2, base_nf * 2,
                    base_nf * 4, base_nf * 4, base_nf * 8, base_nf * 8,
                    base_nf * 8, base_nf * 8]
        layers = []
        for index in range(10):
            layers.append(B.conv_block(
                channels[index], channels[index + 1],
                kernel_size=3 if index % 2 == 0 else 4,
                stride=1 if index % 2 == 0 else 2,
                norm_type=None if index == 0 else norm_type,
                act_type=act_type, mode=mode))
        self.features = B.sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(base_nf * 8 * 8 * 8, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1))

    def forward(self, image):
        feature = self.features(image)
        return self.classifier(feature.flatten(1))


class VGGFeatureExtractor(nn.Module):
    """Frozen ImageNet VGG-19 features through conv5_4 (before ReLU)."""

    def __init__(self, device):
        super(VGGFeatureExtractor, self).__init__()
        try:
            weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
            model = torchvision.models.vgg19(weights=weights)
        except AttributeError:  # torchvision < 0.13
            model = torchvision.models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features.children())[:35])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        self.features.requires_grad_(False)
        self.eval()

    def forward(self, image):
        return self.features((image - self.mean) / self.std)
