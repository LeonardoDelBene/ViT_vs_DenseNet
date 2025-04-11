import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.optim import AdamW, Adam
from torchsummary import summary
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
import os
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from main.py import UMMDSDataset




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

    batch_size = 128
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 0.1
    img_size = 224
    patch_size = 16
    n_classes = 2
    embed_dim = 96+96+96
    depth = 8
    n_heads = 6

    root_dir = '/tmp/Deep_Learning/.venv/UMMDS'

    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7514, 0.5555, 0.6208], std=[0.0395, 0.1003, 0.0496])
    ])

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
    train_dataset, val_dataset, test_dataset = random_split(full_train_dataset, [train_size, val_size, test_size])



    # Creazione dei DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = VisionTransformer(img_size=img_size, patch_size=patch_size, in_chans=3, n_classes=n_classes, embed_dim=embed_dim, depth=depth, n_heads=n_heads)
    model.to(device)
    summary(model, (3, 224, 224))

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint_path = "checkpoint_vit.pth"

    start_epoch = 0
    resume = True

    if resume  and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
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


