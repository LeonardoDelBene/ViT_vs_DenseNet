import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.optim import AdamW, Adam
from torchsummary import summary
from torchvision import datasets
import os
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score


class UMMDSDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.image_paths = []
        self.labels = []

        split_dir = os.path.join(root_dir, split)
        positive_dir = os.path.join(split_dir, "GP")
        negative_dir = os.path.join(split_dir, "GN")

        if os.path.isdir(positive_dir):
            for img_file in os.listdir(positive_dir):
                img_path = os.path.join(positive_dir, img_file)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(1)
        else:
            print(f"Positive directory not found: {positive_dir}")

        if os.path.isdir(negative_dir):
            for img_file in os.listdir(negative_dir):
                img_path = os.path.join(negative_dir, img_file)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(0)
        else:
            print(f"Negative directory not found: {negative_dir}")

        print(f"[{split}] Found {len(self.image_paths)} images: {len(self.labels)} labels (Pos: {self.labels.count(1)}, Neg: {self.labels.count(0)})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Ottieni percorso immagine e label
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Carica l'immagine
        image = Image.open(img_path).convert("RGB")

        # Applica le trasformazioni, se definite
        if self.transform:
            image = self.transform(image)

        return image, label



class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleNeck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([x, out], 1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(BottleNeck(in_channels + i * growth_rate, growth_rate))

        self.denseBlock = nn.Sequential(*layers)

    def forward(self, x):
        return self.denseBlock(x)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, num_block: list, num_classes, growth_rate: int = 32):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        self.conv1 = nn.Conv2d(3, 2 * growth_rate, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(2 * growth_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        num_channels = 2 * growth_rate
        self.dense1 = DenseBlock(num_block[0], num_channels, growth_rate)
        num_channels += num_block[0] * growth_rate
        self.trans1 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2

        self.dense2 = DenseBlock(num_block[1], num_channels, growth_rate)
        num_channels += num_block[1] * growth_rate
        self.trans2 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2

        self.dense3 = DenseBlock(num_block[2], num_channels, growth_rate)
        num_channels += num_block[2] * growth_rate
        self.trans3 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2

        self.dense4 = DenseBlock(num_block[3], num_channels, growth_rate)
        num_channels += num_block[3] * growth_rate

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        x = self.dense1(x)
        x = self.trans1(x)

        x = self.dense2(x)
        x = self.trans2(x)

        x = self.dense3(x)
        x = self.trans3(x)

        x = self.dense4(x)

        x = F.relu(self.bn2(x))

        x = F.adaptive_avg_pool2d(x, (1, 1))

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn')


    root_dir = '/tmp/Deep Learning/.venv/UMMDS'

    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7514, 0.5555, 0.6208], std=[0.0395, 0.1003, 0.0496])
    ])


    train_dataset = UMMDSDataset(root_dir, 'train', transform=transform)
    test_dataset = UMMDSDataset(root_dir, 'test', transform=transform)


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    num_block = [3, 6, 12, 8]
    growth_rate = 32
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet(num_block, num_classes, growth_rate).to(device)

    # Print model summary
    summary(model, (3, 224, 224))

    # Define the optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    #optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 100

    checkpoint_path = "checkpoint.pth"

    # Se esiste un checkpoint, caricalo
    start_epoch = 0
    resume_training = False  # Cambia in True se vuoi riprendere

    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0  # Ricomincia da zero
        print("Starting training from scratch")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint salvato all'epoca {epoch + 1}")

    # Test
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_loss = running_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')

    print("\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")


