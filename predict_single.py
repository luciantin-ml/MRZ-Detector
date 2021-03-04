import torch
import albumentations as A
import torchvision
from albumentations.pytorch import ToTensorV2
from model import UNET
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

IMAGE_HEIGHT = 160*3  # 1280 originally
IMAGE_WIDTH = 240*3  # 1918 originally

train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

device = "cpu"

image = np.array(Image.open('19.png').convert("RGB"))
mask = np.array(Image.open('19.bmp').convert("L"), dtype=np.float32)
mask[mask == 255.0] = 1
augmentations = val_transforms(image=image, mask=mask)
image = augmentations["image"]
mask = augmentations["mask"]

plt.imshow(image.squeeze().permute(1, 2, 0))
plt.show()
plt.imshow(mask, cmap='gray')
plt.show()

image = torch.tensor(image, requires_grad=True).to(device)
image = image.unsqueeze(0)

model = UNET(in_channels=3, out_channels=1).to(device)
print("=> Loading checkpoint")

model.load_state_dict(torch.load("check_Unet_99_95.pth.tar", map_location=torch.device('cpu'))["state_dict"])
# image = image.to(device=device)
model.eval()
with torch.no_grad():
    preds = torch.sigmoid(model(image))
    preds = (preds > 0.5).float()
torchvision.utils.save_image(preds, "./pred_100.png")
model.train()



