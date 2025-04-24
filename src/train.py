from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report

from src.model import LungResNetClassifier
from src.data import load_lung_data
from src.utils import FocalLoss


"""
    Future improvement: Train resnet with a low learning rate.
"""


def train_lung_model(model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     focal: bool = False,
                     num_epochs: int = 10,
                     lr: float = 1e-3,
                     weight_decay: float = 1e-5,
                     class_weights: torch.Tensor = None):
    """
    Train the lung cancer classifier model with tqdm progress bar.

    Args:
        model: The LungResNetClassifier instance.
        train_loader: Dataloader for training data.
        val_loader: Dataloader for validation data.
        num_epochs: Number of epochs to train.
        lr: Learning rate.
        weight_decay: Optimizer weight decay.
        class_weights: Optional weight tensor for imbalanced loss.
    """
    device = model.device
    if not focal:
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    else:
        criterion = FocalLoss(weight=class_weights).to(device)
    
    optimizer = optim.Adam(model.model.fc.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({'loss': total_loss / (total or 1), 'acc': 100 * correct / (total or 1)})

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss:.4f} | Acc: {100 * correct / total:.2f}%")
        evaluate_model(model, val_loader)
    return model


def evaluate_model(model, dataloader):
    """
    Evaluate model and print classification report.
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print(classification_report(y_true, y_pred, target_names=["normal", "adenocarcinoma", "scc"]))



if __name__ == '__main__':
    model = LungResNetClassifier(3)
    # Load data
    train_loader, val_loader, test_loader = load_lung_data(
        path="data",  
        transform=model.transform,
        batch_size=256,
    )
    # Higher weight for malign, with special importance to adenocarcinom
    class_weights = torch.tensor([1.0, 4¡.0, 1.0]).to(model.device) 
    # Train the model
    model = train_lung_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        lr=1e-4,
        weight_decay=1e-5,
        class_weights=class_weights,
        focal = True
    )
    
    print('Model performace on test dataset: ')
    evaluate_model(model, test_loader)