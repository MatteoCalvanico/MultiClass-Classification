import torch
import torch.nn as nn
import torchvision.transforms as transforms

class Net(nn.Module):
    def __init__(self, classes: list[str], dropout: float = 0.5) -> None:
        super(Net, self).__init__()
        
        # Wider architecture with more filters per layer
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=11, stride=4, padding=2),  # 2x wider
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(128, 384, kernel_size=5, padding=2),  # 2x wider
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(384, 768, kernel_size=3, padding=1),  # 2x wider
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 512, kernel_size=3, padding=1),  # 2x wider
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Wider fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512 * 6 * 6, 8192),  # 2x wider
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(8192, 8192),  # 2x wider
            nn.ReLU(inplace=True),
            nn.Linear(8192, len(classes)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_transforms(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])