import os
import numpy as np
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from model import UNET

model = UNET(in_channels=3, out_channels=1)

val_transforms = A.Compose(
    [
        A.Resize(height=480, width=720),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)


class MRZ_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".png", ".bmp"))  ######
        # print(mask_path)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


ds = MRZ_Dataset(
    image_dir="data/train_images/",
    mask_dir="data/train_masks/",
    transform=val_transforms,
)


checkpoint = torch.load("check_Unet_99_95.pth.tar", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["state_dict"])


traced_script_module = torch.jit.trace(model, ds[0][0].unsqueeze(0))

# traced_script_module.save("check_Unet_99_95.pt")
m = torch.jit.script(model)
torch.jit.save(m, 'jitModule.pt')

# plt.imshow(ds[0][0].permute(1,2,0))
# plt.show()
#
# dummy_input = torch.randn(10, 3, 224, 224, device='cpu')
# print(dummy_input.shape)



