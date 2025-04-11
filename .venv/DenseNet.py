import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torch.optim import AdamW, Adam
from torchsummary import summary
from torchvision import datasets
import os
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score


class UMMDSDataset(Dataset):
    def __init__(self, root_dir, num_augmentations, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_augmentations = num_augmentations

        self.image_paths = []
        self.labels = []

        split_dir = os.path.join(root_dir, "train")
        positive_dir = os.path.join(split_dir, "GP")
        negative_dir = os.path.join(split_dir, "GN")

        if os.path.isdir(positive_dir):
            for img_file in os.listdir(positive_dir):
                img_path = os.path.join(positive_dir, img_file)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(1)

        if os.path.isdir(negative_dir):
            for img_file in os.listdir(negative_dir):
                img_path = os.path.join(negative_dir, img_file)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(0)

        print(f"[{split}] Found {len(self.image_paths)} images (Pos: {self.labels.count(1)}, Neg: {self.labels.count(0)})")

    def __len__(self):
        return len(self.image_paths) * self.num_augmentations  # Ogni immagine viene usata 5 volte

    def __getitem__(self, idx):
        img_idx = idx // self.num_augmentations  # Ripete ogni immagine 5 volte
        img_path = self.image_paths[img_idx]
        label = self.labels[img_idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)  # Ogni volta che viene chiamato, il crop è casuale

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


    root_dir = '/tmp/Deep_Learning/.venv/UMMDS'

    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7514, 0.5555, 0.6208], std=[0.0395, 0.1003, 0.0496])
    ])

    num_block = [6, 12, 24, 16]
    growth_rate = 32
    num_classes = 2
    num_epochs = 50
    batch_size = 64

    num_argumentations = 1
    dataset = UMMDSDataset(root_dir, num_argumentations, transform=transform)

    # Definiamo la dimensione del train e validation set
    dataset_length = len(dataset)

    # Compute sizes
    train_size = int(0.8 * dataset_length)
    val_size = int(0.1 * dataset_length)
    test_size = dataset_length - train_size - val_size  # Ensure the sum is correct

    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Creazione dei DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet(num_block, num_classes, growth_rate).to(device)

    # Print model summary
    summary(model, (3, 224, 224))

    # Define the optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()


    checkpoint_path = "checkpoint.pth"

    # Se esiste un checkpoint, caricalo
    start_epoch = 0
    resume_training = True  # Cambia in True se vuoi riprendere

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

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total


        # Salva il checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
        }
        torch.save(checkpoint, checkpoint_path)

        # Validation ogni 5 epoche
        if (epoch + 1) % 1 == 0:
            model.eval()
            running_loss_val = 0.0
            correct_val = 0
            total_val = 0

            all_labels_val = []
            all_predictions_val = []

            with torch.no_grad():
                for inputs_val, labels_val in val_loader:
                    inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                    outputs_val = model(inputs_val)
                    loss_val = criterion(outputs_val, labels_val)

                    running_loss_val += loss_val.item()
                    _, predicted_val = torch.max(outputs_val, 1)
                    total_val += labels_val.size(0)
                    correct_val += (predicted_val == labels_val).sum().item()

                    all_labels_val.extend(labels_val.cpu().numpy())
                    all_predictions_val.extend(predicted_val.cpu().numpy())

            val_loss = running_loss_val / len(val_loader)
            val_accuracy = 100 * correct_val / total_val

            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')


    print("Training completed")

    model.eval()
    running_loss_test = 0.0
    correct_test = 0
    total_test = 0

    all_labels_test = []
    all_predictions_test = []

    with torch.no_grad():
        for inputs_test, labels_test in test_loader:
            inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
            outputs_test = model(inputs_test)
            loss_test = criterion(outputs_test, labels_test)

            running_loss_test += loss_test.item()
            _, predicted_test = torch.max(outputs_test, 1)
            total_test += labels_test.size(0)
            correct_test += (predicted_test == labels_test).sum().item()

            all_labels_test.extend(labels_test.cpu().numpy())
            all_predictions_test.extend(predicted_test.cpu().numpy())

    test_loss = running_loss_test / len(test_loader)
    test_accuracy = 100 * correct_test / total_test

    f1 = f1_score(all_labels_test, all_predictions_test, average='weighted')
    precision = precision_score(all_labels_test, all_predictions_test, average='weighted')
    recall = recall_score(all_labels_test, all_predictions_test, average='weighted')

    print("\nTest Results after training")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")


