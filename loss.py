import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import one_hot_embedding


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self, x, y):

        t = torch.nn.functional.one_hot(
            y.long(), num_classes=1 + self.num_classes
        ).float()
        t = t[:, 1:]
        # BCE_loss = F.binary_cross_entropy_with_logits(x, t, reduction="none")
        # pt = torch.exp(-BCE_loss)  # 计算 p_t
        # focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        # return torch.sum(focal_loss)

        p = torch.sigmoid(x)
        ce_loss = F.binary_cross_entropy_with_logits(x, t, reduction="none")
        p_t = p * t + (1 - p) * (1 - t)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * t + (1 - self.alpha) * (1 - t)
            loss = alpha_t * loss

        return torch.sum(loss)

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        """
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)  # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(
            masked_loc_preds, masked_loc_targets, reduction="sum"
        )

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])

        # print(
        #     "loc_loss: %.3f | cls_loss: %.3f"
        #     % (loc_loss.item() / num_pos, loc_loss.item() / num_pos),
        #     end=" | ",
        # )
        if num_pos == 0:
            return cls_loss
        else:
            return loc_loss / num_pos + cls_loss / num_pos
