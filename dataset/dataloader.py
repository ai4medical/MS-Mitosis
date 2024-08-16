import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml
import os
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from .encoder import DataEncoder
from .transform import random_crop, random_flip, resize
from utils import change_box_order
import random


def yolobox_converter(line: str):
    box = line.strip().split(" ")
    return (
        int(float(box[1]) * 512),
        int(float(box[2]) * 512),
        int(float(box[3]) * 512),
        int(float(box[4]) * 512),
        int(box[0]),
    )


class MitosisDataset(Dataset):
    def __init__(self, data_root: str) -> None:
        super().__init__()

        self.encoder = DataEncoder()

        self.data_root = data_root
        self.image_files = []
        self.modalities = []
        for modality in os.listdir(self.data_root):
            cur_image_files = os.listdir(
                os.path.join(self.data_root, modality, "images")
            )
            self.image_files += cur_image_files
            self.modalities += [int(modality[3:])] * len(cur_image_files)

        self.transforms = T.Compose([T.Resize((512, 512)), T.ToTensor()])

    def __getitem__(self, index):
        modality = self.modalities[index]
        image = Image.open(
            os.path.join(
                self.data_root, f"mod{modality}", "images", self.image_files[index]
            )
        )
        boxes = []
        labels = []
        if os.path.exists(
            label_path := os.path.join(
                self.data_root,
                f"mod{modality}",
                "labels",
                self.image_files[index].replace(".jpg", ".txt"),
            )
        ):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    line_splits = line.strip().split(" ")
                    boxes.append([int(float(s) * 512) for s in line_splits[1:]])
                    labels.append(int(line_splits[0]))
        if len(boxes):
            boxes = change_box_order(torch.FloatTensor(boxes), "xywh2xyxy")
        labels = torch.FloatTensor(labels)
        image, boxes = random_flip(image, boxes)
        image, boxes = random_crop(image, boxes)
        image, boxes = resize(image, boxes, (512, 512))
        image = TF.to_tensor(image)
        if len(boxes):
            boxes = change_box_order(boxes, "xyxy2xywh")
        image = TF.adjust_brightness(image, 1 + (random.random() - 0.5) * 0.2)
        image = TF.adjust_contrast(image, 1 + (random.random() - 0.5) * 0.2)
        image = TF.adjust_hue(image, (random.random() - 0.5) * 0.2)
        return {
            "image": image,
            "box": boxes,
            "label": labels,
            "modality": modality,
        }

    def __len__(self):
        return len(self.image_files)

    def collate_fn(self, batch):
        """Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        """
        imgs = [x["image"] for x in batch]
        boxes = [x["box"] for x in batch]
        labels = [x["label"] for x in batch]
        modalities = [x["modality"] for x in batch]
        # print(modalities)

        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, 512, 512)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(
                boxes[i], labels[i], input_size=(512, 512)
            )
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return {
            "image": inputs,
            "box": torch.stack(loc_targets),
            "label": torch.stack(cls_targets),
            "modality": torch.FloatTensor(modalities),
        }


class MitosisEvalDataset(Dataset):
    def __init__(self, data_root: str) -> None:
        super().__init__()

        self.encoder = DataEncoder()

        self.data_root = data_root
        self.image_files = []
        self.image_files = os.listdir(os.path.join(self.data_root, "images"))

        self.transforms = T.Compose([T.Resize((512, 512)), T.ToTensor()])

    def __getitem__(self, index):
        image = Image.open(
            os.path.join(self.data_root, "images", self.image_files[index])
        )
        boxes = []
        labels = []
        if os.path.exists(
            label_path := os.path.join(
                self.data_root,
                "labels",
                self.image_files[index].replace(".jpg", ".txt"),
            )
        ):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    line_splits = line.strip().split(" ")
                    boxes.append([int(float(s) * 512) for s in line_splits[1:]])
                    labels.append(int(line_splits[0]))
        if len(boxes):
            boxes = change_box_order(torch.FloatTensor(boxes), "xywh2xyxy")
        labels = torch.FloatTensor(labels)
        image, boxes = resize(image, boxes, (512, 512))
        image = TF.to_tensor(image)
        return {
            "image": image,
            "box": boxes,
            "label": labels,
        }

    def __len__(self):
        return len(self.image_files)

    def collate_fn(self, batch):
        """Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        """
        imgs = [x["image"] for x in batch]
        boxes = [x["box"] for x in batch]
        labels = [x["label"] for x in batch]

        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, 512, 512)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(
                boxes[i], labels[i], input_size=(512, 512)
            )
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return {
            "image": inputs,
            "box": torch.stack(loc_targets),
            "label": torch.stack(cls_targets),
        }


if __name__ == "__main__":
    dataset = MitosisDataset("dataset/MIDOG.yml", "train")
    dataloader = DataLoader(dataset, batch_size=512, collate_fn=dataset.collate_fn)
    datapack = next(iter(dataloader))

    images = datapack["image"]
    annotations = datapack["annotation"]
    modalities = datapack["modality"]
