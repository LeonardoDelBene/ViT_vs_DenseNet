import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim import AdamW
from torchview import draw_graph
from torchsummary import summary
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from  sklearn.metrics import f1_score, precision_score, recall_score


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

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)


    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (q @ k_t) * self.scale  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, n_classes=2, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))

        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)


    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)

        return x


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn')

    batch_size = 64
    num_epochs = 100
    learning_rate = 1e-4
    img_size = 224
    patch_size = 16
    n_classes = 2
    embed_dim = 768
    depth = 12
    n_heads = 12

    root_dir = '/tmp/Deep Learning/.venv/UMMDS'

    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7514, 0.5555, 0.6208], std=[0.0395, 0.1003, 0.0496])
    ])


    train_dataset = UMMDSDataset(root_dir, 'train', transform=transform)
    test_dataset = UMMDSDataset(root_dir, 'test', transform=transform)


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = VisionTransformer(img_size=img_size, patch_size=patch_size, in_chans=3, n_classes=n_classes, embed_dim=embed_dim, depth=depth, n_heads=n_heads)
    model.to(device)
    summary(model, (3, 224, 224))

    #optimizer = Adam(model.parameters(), lr=learning_rate)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0.0
    num_epochs = 100

    checkpoint_path = "checkpoint_vit.pth"

    start_epoch = 0
    resume = True

    if resume  and os.path.exists(checkpoint_path):
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
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

        # Salva il checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint salvato all'epoca {epoch + 1}")

    print("Training finished.")
    print("Start testing.")
    # Test
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []  # Per memorizzare tutte le etichette reali
    all_predictions = []  # Per memorizzare tutte le previsioni del modello

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()  # Inizia la misurazione del tempo

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

    end_event.record()  # Fine misurazione del tempo

    torch.cuda.synchronize()  # Assicura che tutti gli eventi siano completati

    total_time = start_event.elapsed_time(end_event) / 1000  # Tempo totale in secondi
    fps = total / total_time  # Numero di immagini elaborate al secondo

    test_loss = running_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    f1 = f1_score(all_labels, all_predictions, average='weighted')  # Weighted: tiene conto della classe bilanciata
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')

    print(f"Test Results: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"FPS (Frames Per Second): {fps:.2f}")






