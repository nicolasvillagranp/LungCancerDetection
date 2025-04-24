import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image

class LungCancerDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.samples = []
        self.labels = []
        self.class_to_idx = {"lung_n": 0, "lung_aca": 1, "lung_scc": 2}
        self.transform = transform

        for class_name, class_idx in self.class_to_idx.items():
            class_path = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_path):
                if fname.endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(os.path.join(class_path, fname))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image = read_image(self.samples[index]).float()
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_lung_data(path: str, transform, batch_size: int = 256):
    dataset = LungCancerDataset(path, transform=transform)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size],
                                                generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader