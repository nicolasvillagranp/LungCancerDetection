import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class LungResNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True, device: str = None):
        super().__init__()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = models.resnet50(pretrained=pretrained)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace and unfreeze classifier layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(in_features, 8 * in_features),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(8 * in_features, num_classes)
                                      )


        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img / 255.0)  # normalize to [0, 1] first
