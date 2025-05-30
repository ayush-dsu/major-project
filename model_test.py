# predict_raster.py

import geopandas as gpd
import rasterio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from rasterio import features
from shapely.geometry import mapping
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------
# Paths and Parameters
# -----------------------
new_image_path = "Materials/raster/2024/LC09_L2SP_144051_20241231_20250102_02_T1_SR__stack_raster.tif"  # <- Change this for new image
road_shapefile = "road_nh_948.shp"
model_path = "model/lulc_resnet18_and_mobilenetv2_combo.pth"
classified_output = "classified_output_new.tif"
buffer_distance = 1000  # meters

# -----------------------
# Load Raster and Road
# -----------------------
with rasterio.open(new_image_path) as src:
    image = src.read([4, 3, 2])
    meta = src.meta.copy()
    transform = src.transform
    crs = src.crs
    shape = (src.height, src.width)

road = gpd.read_file(road_shapefile).to_crs(crs)
buffer = road.copy()
buffer['geometry'] = road.geometry.buffer(buffer_distance)

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
        return patch, 0  # dummy label

# -----------------------
# Mask Buffer Area
# -----------------------
mask = features.geometry_mask([mapping(geom) for geom in buffer.geometry],
                              transform=transform, invert=True, out_shape=shape)
coords = np.column_stack(np.where(mask))
dummy_labels = np.zeros(len(coords))
dataset = PatchDataset(image, coords, dummy_labels)
loader = DataLoader(dataset, batch_size=256)

# -----------------------
# Load Model
# -----------------------
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# model = resnet18(weights=ResNet18_Weights.DEFAULT)
# model.fc = nn.Linear(model.fc.in_features, 6)
# from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
# model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
# model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 6)

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
    
model = HybridResNetMobileNet(num_classes=6)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# -----------------------
# Predict
# -----------------------
print("Classifying...")
preds = []
for images, _ in tqdm(loader, desc="Predicting"):
    images = images.to(device)
    outputs = model(images)
    preds.extend(outputs.argmax(dim=1).cpu().numpy())

classified = np.zeros(shape, dtype=np.uint8)
classified[coords[:, 0], coords[:, 1]] = preds

# -----------------------
# Save Result
# -----------------------
meta.update({"count": 1, "dtype": "uint8"})
with rasterio.open(classified_output, 'w', **meta) as dst:
    dst.write(classified, 1)

print(f"Saved classified image to: {classified_output}")

# Optional: Visualization
labels_list = ['Background', 'Water Bodies', 'Urban Land', 'Vegetation', 'Barren Land', 'Agricultural Land']
unique, counts = np.unique(preds, return_counts=True)
class_counts = dict(zip(unique, counts))
counts_full = [class_counts.get(i, 0) for i in range(6)]

fig, axs = plt.subplots(1,2, figsize=(12,5))
axs[0].bar(labels_list, counts_full)
axs[0].set_title("Bar Chart")
axs[0].tick_params(axis='x', rotation=45)
axs[1].pie(counts_full, labels=labels_list, autopct='%1.1f%%')
axs[1].set_title("Pie Chart")
plt.tight_layout()
plt.show()