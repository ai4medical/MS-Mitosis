import torch
import torch.nn as nn

from .fpn import FPN
from torchvision.models.resnet import *
from ..gradient_reversal import RevGrad
from loss import FocalLoss


class RetinaNet(nn.Module):
    num_anchors = 9

    def __init__(
        self,
        resnet_size: int,
        num_classes: int,
        num_headers: int,
        criterion=None,
        gradient_reversal_factor=0.02,
        _lambda=0.2,
    ):
        super(RetinaNet, self).__init__()
        self.fpn = FPN(resnet_size)
        self.num_classes = num_classes
        self.num_headers = num_headers
        self.loc_heads = nn.ModuleList(
            [self._make_head(self.num_anchors * 4) for _ in range(num_headers)]
        )
        self.cls_heads = nn.ModuleList(
            [
                self._make_head(self.num_anchors * self.num_classes)
                for _ in range(num_headers)
            ]
        )

        self.criterion = criterion
        self._lambda = _lambda

        self.gradient_reversal = RevGrad(gradient_reversal_factor)

    def forward(self, x, modality=None, loc_targets=None, cls_targets=None):
        if self.num_headers == 1:
            if not self.training:
                fms = self.fpn(x)
                loc_preds = []
                cls_preds = []
                for fm in fms:
                    loc_pred = self.loc_heads[0](fm)
                    cls_pred = self.cls_heads[0](fm)
                    loc_pred = (
                        loc_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
                    )
                    cls_pred = (
                        cls_pred.permute(0, 2, 3, 1)
                        .contiguous()
                        .view(x.size(0), -1, self.num_classes)
                    )
                    loc_preds.append(loc_pred)
                    cls_preds.append(cls_pred)
                return torch.stack([torch.cat(loc_preds, 1)], dim=1), torch.stack(
                    [torch.cat(cls_pred, 1)], dim=1
                )
            else:
                fms = self.fpn(x)
                loc_preds = []
                cls_preds = []
                for fm in fms:
                    loc_pred = self.loc_heads[0](fm)
                    cls_pred = self.cls_heads[0](fm)
                    loc_pred = (
                        loc_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
                    )
                    cls_pred = (
                        cls_pred.permute(0, 2, 3, 1)
                        .contiguous()
                        .view(x.size(0), -1, self.num_classes)
                    )
                    loc_preds.append(loc_pred)
                    cls_preds.append(cls_pred)
                loc_preds = torch.cat(loc_preds, 1)
                cls_preds = torch.cat(cls_preds, 1)
                loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
                return loss
        else:
            if not self.training:
                all_header_loc_preds = []
                all_header_cls_preds = []
                fms = self.fpn(x)
                for k in range(len(self.loc_heads)):
                    loc_preds = []
                    cls_preds = []
                    for fm in fms:
                        loc_pred = self.loc_heads[k](fm)
                        cls_pred = self.cls_heads[k](fm)
                        loc_pred = (
                            loc_pred.permute(0, 2, 3, 1)
                            .contiguous()
                            .view(x.size(0), -1, 4)
                        )
                        cls_pred = (
                            cls_pred.permute(0, 2, 3, 1)
                            .contiguous()
                            .view(x.size(0), -1, self.num_classes)
                        )
                        loc_preds.append(loc_pred)
                        cls_preds.append(cls_pred)
                    all_header_loc_preds.append(torch.cat(loc_preds, 1))
                    all_header_cls_preds.append(torch.cat(cls_preds, 1))
                return torch.stack(all_header_loc_preds, dim=1), torch.stack(
                    all_header_cls_preds, dim=1
                )
            else:
                modality_grouped_x = [
                    x[modality == m] for m in range(len(self.loc_heads))
                ]
                modality_grouped_loc_targets = [
                    loc_targets[modality == m] for m in range(len(self.loc_heads))
                ]
                modality_grouped_cls_targets = [
                    cls_targets[modality == m] for m in range(len(self.loc_heads))
                ]
                total_loss = 0
                for m, (x, loc_target, cls_target) in enumerate(
                    zip(
                        modality_grouped_x,
                        modality_grouped_loc_targets,
                        modality_grouped_cls_targets,
                    )
                ):
                    if not x.size(0):
                        continue

                    fms = self.fpn(x)
                    losses = []
                    for k in range(len(self.loc_heads)):
                        loc_preds = []
                        cls_preds = []
                        for fm in fms:
                            if m != k:
                                fm = self.gradient_reversal(fm)
                            loc_pred = self.loc_heads[k](fm)
                            cls_pred = self.cls_heads[k](fm)
                            loc_pred = (
                                loc_pred.permute(0, 2, 3, 1)
                                .contiguous()
                                .view(x.size(0), -1, 4)
                            )
                            cls_pred = (
                                cls_pred.permute(0, 2, 3, 1)
                                .contiguous()
                                .view(x.size(0), -1, self.num_classes)
                            )
                            loc_preds.append(loc_pred)
                            cls_preds.append(cls_pred)
                        loc_preds = torch.cat(loc_preds, 1)
                        cls_preds = torch.cat(cls_preds, 1)
                        losses.append(
                            self.criterion(loc_preds, loc_target, cls_preds, cls_target)
                        )
                    loss = 0
                    for k, l in enumerate(losses):
                        if k != m:
                            loss += l * self._lambda
                    loss /= len(self.loc_heads) - 1
                    loss += losses[m]
                    total_loss += loss
                return total_loss

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)
