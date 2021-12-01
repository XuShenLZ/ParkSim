import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(pred, gt, batch_average=True):
    """
    penalty-reduced pixel-wise logistic regression with focal loss
    """
    eps = 1e-7

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred - eps) * torch.pow(pred, 2) * neg_weights * neg_inds

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if batch_average:
        batch_size = gt.shape[0]
        loss = -(pos_loss + neg_loss) / batch_size
    else:
        loss = -(pos_loss + neg_loss)

    return loss

def offset_loss(pred, gt, batch_average=True):
    """
    L1 loss for offset prediction
    """
    gt_total = gt[:, 0, :, :] + gt[:, 1, :, :]
    mask = gt_total.gt(0).float()

    loss = F.l1_loss(pred[:, 0, :, :] * mask, gt[:, 0, :, :], reduction='sum') \
        + F.l1_loss(pred[:, 1, :, :] * mask, gt[:, 1, :, :], reduction='sum')

    if batch_average:
        batch_size = gt.shape[0]
        loss = loss / batch_size

    return loss

class FullyConvLoss(nn.Module):
    """
    Loss: focal and l1 loss
    """
    def __init__(self, lam=1):
        """
        instantiate
        """
        super(FullyConvLoss, self).__init__()
        self.lam = lam
        self.focal_loss = focal_loss
        self.offset_loss = offset_loss

    def forward(self, pred, gt):
        """
        forward method
        """
        heatmap_pred = pred[:, 0, :, :]
        offset_pred = pred[:, 1:, :, :]

        heatmap_gt = gt[:, 0, :, :]
        offset_gt = gt[:, 1:, :, :]

        return self.focal_loss(heatmap_pred, heatmap_gt) + self.lam * self.offset_loss(offset_pred, offset_gt)
