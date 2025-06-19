import torch
import os
import cv2


class PetDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: str, mask_folder: str = "trimaps", train: bool = True, transform: Optional[Callable] = None):
        super().__init__()
        self.image_dir = os.path.join(data_root, "images")
        self.annotations_dir = os.path.join(data_root, "annotations")
        self.mask_folder = mask_folder

        if train:
            with open(os.path.join(self.annotations_dir, "trainval.txt")) as rf:
                all_data = rf.readlines()
        else:
            with open(os.path.join(self.annotations_dir, "test.txt")) as rf:
                all_data = rf.readlines()

        self.transform = transform
        self.image_names = [row.split()[0] for row in all_data]
        self.image_classes = [1 if image_name[0].isupper() else 2 for image_name in self.image_names]


    def __len__(self) -> int:
        return len(self.image_names)


    def __getitem__(self, idx: int) -> torch.Tensor:
        image_name = self.image_names[idx]
        full_image_path = os.path.join(self.image_dir, f"{image_name}.jpg")
        full_mask_path = os.path.join(self.annotations_dir, self.mask_folder, f"{image_name}.png")

        image = cv2.imread(full_image_path)[..., ::-1]
        mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask_folder == "trimaps":
            mask[mask == 2] = 0
            mask[(mask == 1) | (mask == 3)] = self.image_classes[idx]

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image'] / 255.
            mask = transformed['mask'].long()

        return image, mask
