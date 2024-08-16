from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model.retinanet import RetinaNet
import yaml
import sys
from typing import *
from dataset.dataloader import MitosisDataset
from loss import FocalLoss
from datetime import datetime
import json


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0
        self.num = 0

    def add(self, value, num=1):
        self.sum += value
        self.num += num

    def get(self) -> float:
        if self.num:
            return self.sum / self.num
        else:
            return 0

    def clear(self) -> None:
        self.sum = 0
        self.num = 0

    def __str__(self) -> str:
        return str(self.get())

    def __repr__(self) -> str:
        return str(self.get())


def main(config: Dict) -> None:

    log_dir = os.path.join(
        config["logdir"],
        config["index"],
        datetime.now().strftime("%m-%d %H:%M:%S"),
    )

    writer = SummaryWriter(log_dir)

    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    criterion = FocalLoss(config["num_classes"], config["alpha"], config["gamma"])

    model = RetinaNet(
        config["resnet_size"],
        config["num_classes"],
        config["num_headers"],
        criterion,
        gradient_reversal_factor=config["gradient_reversal_factor"],
        _lambda=config["lambda"],
    )
    model = model.cuda()
    if config["ckpt"]:
        model.load_state_dict(torch.load(config["ckpt"]))

    train_dataset = MitosisDataset(config["data_root"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        pin_memory=True,
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer=optimizer,
        base_lr=config["lr"] / 10,
        max_lr=config["lr"],
        cycle_momentum=False,
    )
    pbar = tqdm(
        range(1, config["max_iters"]),
        ncols=80,
        total=config["max_iters"],
    )
    n_iter = 0

    train_loader_iter = iter(train_loader)

    best_f1 = 0

    model.train()
    avg_loss = AverageMeter()
    for n_iter in pbar:
        try:
            datapack = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            datapack = next(train_loader_iter)

        images = datapack["image"].cuda()
        boxes = datapack["box"].cuda()
        labels = datapack["label"].cuda()
        modalities = datapack["modality"].cuda()

        loss = model(images, modalities, boxes, labels)
        avg_loss.add(loss.item())

        pbar.set_postfix({"ls": avg_loss.get()})

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        scheduler.step()

        if n_iter % 100 == 0:
            writer.add_scalar("train/loss", avg_loss.get(), n_iter)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], n_iter)
            avg_loss.clear()

            torch.save(
                model.state_dict(),
                os.path.join(log_dir, "final.pth"),
            )


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    main(config)
