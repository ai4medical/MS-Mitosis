from PIL import Image
import torch
from model.retinanet import RetinaNet
import os
import torchvision.transforms.functional as TF
import json
from dataset.encoder import DataEncoder
from rich import print
from tqdm import tqdm
from torch.utils.data import DataLoader
import random

CROP_SIZE = 512
STRIDE_SIZE = CROP_SIZE


def convert_yolo_line(line: str):
    splits = line.strip().split(" ")
    x_center, y_center, width, height = tuple([float(s) * 4096 for s in splits[1:]])
    return (
        x_center - width / 2,
        y_center - height / 2,
        x_center + width / 2,
        y_center + height / 2,
    )


def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    intersection_area = max(0, x2_inter - x1_inter + 1) * max(
        0, y2_inter - y1_inter + 1
    )

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area

    return iou


def cauculate_metrics(pred_boxes, pred_labels, target_boxes, target_labels):
    pred_boxes = [
        pred_box for i, pred_box in enumerate(pred_boxes) if pred_labels[i] == 0
    ]
    pred_labels = [
        pred_label for i, pred_label in enumerate(pred_labels) if pred_label == 0
    ]

    TP = 0
    FP = 0
    FN = 0
    for pred_box, pred_label in zip(pred_boxes, pred_labels):
        matched_flag = False
        for i, (target_box, target_label) in enumerate(
            zip(target_boxes, target_labels)
        ):
            if calculate_iou(pred_box, target_box) > 0.5 and pred_label == target_label:
                TP += 1
                matched_flag = True
                del target_boxes[i]
                break
        if not matched_flag:
            FP += 1
    FN = len(target_boxes)
    return TP, FP, FN


def is_in(inner_box: tuple[int, int, int, int], outter_box: tuple[int, int, int, int]):
    return (
        inner_box[0] >= outter_box[0]
        and inner_box[1] >= outter_box[1]
        and inner_box[2] <= outter_box[2]
        and inner_box[3] <= outter_box[3]
    )


def evaluate(
    ckpt_path: str,
    resnet_size: int,
    num_classes: int,
    num_headers: int,
):

    DATA_ROOT = "your data root"

    model = RetinaNet(resnet_size, num_classes, num_headers).cuda()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    data_encoder = DataEncoder()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    sample_names = [
        os.path.splitext(fname)[0]
        for fname in os.listdir(os.path.join(DATA_ROOT, "labels"))
    ]
    for sample_name in tqdm(sample_names, ncols=80):
        whole_image = Image.open(
            os.path.join(DATA_ROOT, "images", sample_name + ".jpg")
        )
        with open(
            os.path.join(DATA_ROOT, "labels-phony-2", sample_name + ".txt"), "r"
        ) as f:
            gt_boxes = [convert_yolo_line(line) for line in f.readlines()]
        whole_pred_boxes = []
        whole_pred_labels = []
        cursors = [(xx, yy) for xx in range(0, 4096, 512) for yy in range(0, 4096, 512)]
        image_tensors = torch.stack(
            [
                TF.to_tensor(whole_image.crop((xx, yy, xx + 512, yy + 512)))
                for xx, yy in cursors
            ]
        ).cuda()
        with torch.no_grad():
            loc_preds, cls_preds = model(image_tensors)
        loc_preds, cls_preds = loc_preds.cpu(), cls_preds.cpu()
        for i, (xx, yy) in enumerate(cursors):
            pred_boxes, pred_labels = data_encoder.decode(
                loc_preds[i].mean(dim=0),
                cls_preds[i].mean(dim=0),
                (512, 512),
                CLS_THRESH=0.4,
            )
            pred_boxes, pred_labels = pred_boxes.tolist(), pred_labels.tolist()
            pred_boxes = [
                (box[0] + xx, box[1] + yy, box[2] + xx, box[3] + yy)
                for box in pred_boxes
            ]
            whole_pred_boxes += pred_boxes
            whole_pred_labels += pred_labels

        cur_tp, cur_fp, cur_fn = cauculate_metrics(
            whole_pred_boxes, whole_pred_labels, gt_boxes, [0] * len(gt_boxes)
        )
        total_tp += cur_tp
        total_fp += cur_fp
        total_fn += cur_fn

    P = total_tp / (total_tp + total_fp)
    R = total_tp / (total_tp + total_fn)
    F1 = 2 * P * R / (P + R)
    return P, R, F1


if __name__ == "__main__":
    P, R, F1 = evaluate("tblog/JZL-all/08-15 12:13:46/final.pth", 18, 1, 5)
    print(f"P: {P}")
    print(f"R: {R}")
    print(f"F1: {F1}")
