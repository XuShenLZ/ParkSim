import torch
import torch.nn as nn

def focal_loss(pred, gt, size_average=True):
    """
    penalty-reduced pixel-wise logistic regression with focal loss
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    # if size_average:
    #     batch_size = pred.shape[0]
    #     loss = -(pos_loss + neg_loss) / batch_size
    # else:
    loss = -(pos_loss + neg_loss) / 10.0

    return loss

class IntentNetLoss(nn.Module):
    """
    Loss: focal and l1 loss
    """
    def __init__(self, device, lam=1):
        """
        instantiate
        """
        super(IntentNetLoss, self).__init__()
        self.device = device
        self.lam = lam
        self.focal_loss = focal_loss
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_all, gt_all):
        """
        forward method
        """
        heatmap_pred = pred_all[0].to(self.device)
        offset_pred = pred_all[1].to(self.device)

        heatmap_gt = gt_all[0].to(self.device)
        offset_gt = gt_all[1].to(self.device)

        return self.focal_loss(heatmap_pred, heatmap_gt) + self.l1_loss(offset_pred, offset_gt)
