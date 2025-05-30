# train_model.py

import geopandas as gpd
import rasterio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from rasterio import features
from shapely.geometry import mapping
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

# # -----------------------
# # File Paths
# # -----------------------
nh948_path = "road_nh_948.shp"
training_path = "training_input.shp"
image_path = "Materials/raster/2020/LC08_L2SP_144051_20201212_20210313_02_T1_SR__stack_raster.tif"
model_output_path = "model/lulc_resnet18_and_mobilenetv2_combo.pth"

# -----------------------
# Load Raster and Shapefiles
# -----------------------
with rasterio.open(image_path) as src:
    image = src.read([4, 3, 2])
    meta = src.meta.copy()
    transform = src.transform
    crs = src.crs
    shape = (src.height, src.width)

nh948 = gpd.read_file(nh948_path).to_crs(crs)
training = gpd.read_file(training_path).to_crs(crs)

# -----------------------
# Rasterize Training Labels
# -----------------------
label_raster = features.rasterize(
    ((geom, value) for geom, value in zip(training.geometry, training['class'])),
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype='uint8'
)

# -----------------------
# Dataset Class
# -----------------------
class PatchDataset(Dataset):
    def __init__(self, image, coords, labels, patch_size=32):
        self.image = image
        self.coords = coords
        self.labels = labels
        self.patch_size = patch_size

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        half = self.patch_size // 2
        y1 = np.clip(y - half, 0, self.image.shape[1] - 1)
        y2 = np.clip(y + half, 0, self.image.shape[1])
        x1 = np.clip(x - half, 0, self.image.shape[2] - 1)
        x2 = np.clip(x + half, 0, self.image.shape[2])
        patch = self.image[:, y1:y2, x1:x2]
        pad_y = self.patch_size - (y2 - y1)
        pad_x = self.patch_size - (x2 - x1)
        if pad_y > 0 or pad_x > 0:
            patch = np.pad(patch, ((0,0), (0, pad_y), (0, pad_x)), mode='constant')
        patch = torch.tensor(patch).float() / 255.0
        return patch, self.labels[idx]

# -----------------------
# Prepare Data
# -----------------------
mask = label_raster > 0
coords = np.column_stack(np.where(mask))
labels = label_raster[mask]

train_coords, test_coords, train_labels, test_labels = train_test_split(coords, labels, test_size=0.2, random_state=42)

train_dataset = PatchDataset(image, train_coords, train_labels)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# -----------------------
# Model Setup
# -----------------------
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#ResNet18
# from torchvision.models import resnet18, ResNet18_Weights
# model = resnet18(weights=ResNet18_Weights.DEFAULT)
# model.fc = nn.Linear(model.fc.in_features, 6)
# model = model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#VGG16
# from torchvision.models import vgg16, VGG16_Weights
# model = vgg16(weights=VGG16_Weights.DEFAULT)
# model.classifier[6] = nn.Linear(model.classifier[6].in_features, 6)
# model = model.to(device)

#MobileNetV2
# from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
# model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
# model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 6)
# model = model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ...existing code...

#ResNet18 and MobileNetV2 Combo
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights

class HybridResNetMobileNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1])  # (B, 512, 1, 1)
        self.mobilenet_features = self.mobilenet.features  # (B, 1280, 1, 1)
        self.resnet_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mobilenet_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 + 1280, num_classes)

    def forward(self, x):
        r = self.resnet_pool(self.resnet_features(x))
        r = r.view(r.size(0), -1)
        m = self.mobilenet_pool(self.mobilenet_features(x))
        m = m.view(m.size(0), -1)
        combined = torch.cat([r, m], dim=1)
        out = self.classifier(combined)
        return out

model = HybridResNetMobileNet(num_classes=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# Train Model
# -----------------------
print("Training model...")
for epoch in range(5):
    model.train()
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/5"):
        images, labels = images.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# -----------------------
# Save Model
# -----------------------
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), model_output_path)
print(f"Model saved to {model_output_path}")