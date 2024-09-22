# losses.py
import torch
import torch.nn as nn

class ContrastiveCenterLoss(nn.Module):
    """Contrastive Center Loss.

    This loss combines the concepts of Center Loss and Contrastive Loss,
    considering both intra-class compactness and inter-class separability.

    Parameters:
        num_classes (int): Number of classes.
        feat_dim (int): Dimension of features.
        use_gpu (bool): Whether to use GPU.
        lambda_c (float): Weight of the center loss part.
    """
    def __init__(self, num_classes=2, feat_dim=256, use_gpu=True, lambda_c=1.0):
        super(ContrastiveCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.lambda_c = lambda_c

        # Initialize class centers
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Forward propagation function.

        Parameters:
            x: Feature matrix, shape (batch_size, feat_dim).
            labels: Ground truth labels, shape (batch_size).
        """
        batch_size = x.size(0)

        # Compute distance between each feature and all class centers
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        # Create a mask for the class labels
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        # Compute intra-class distances (distance to the correct class center)
        intra_distances = distmat * mask.float()
        intra_distances = intra_distances.sum() / batch_size

        # Compute inter-class distances (distance to incorrect class centers)
        inter_distances = distmat * (~mask).float()
        inter_distances = inter_distances.sum() / (batch_size * (self.num_classes - 1))

        # Compute the contrastive center loss
        loss = (self.lambda_c / 2.0) * intra_distances / (inter_distances + 1e-6) / 0.1

        return loss

def compute_loss(class_logits, masks, upsampled_embedding, alpha, loss_fn, contrastive_center_loss, ce_weight=1.0, center_weight=1.0):
    """
    Compute cross-entropy loss and contrastive center loss, and combine them with given weights.

    Args:
        class_logits (Tensor): Classification results (B, num_classes, H, W)
        masks (Tensor): Masks (B, H, W)
        upsampled_embedding (Tensor): Upsampled embeddings (B, C, H, W)
        alpha (float): Weight of contrastive center loss
        loss_fn (nn.Module): Cross-entropy loss function
        contrastive_center_loss (ContrastiveCenterLoss): Instance of contrastive center loss function
        ce_weight (float): Weight of cross-entropy loss
        center_weight (float): Weight of contrastive center loss

    Returns:
        Tensor: Total loss
        float: Cross-entropy loss value
        float: Contrastive center loss value
    """
    # Compute cross-entropy loss
    loss_ce = loss_fn(class_logits, masks.long())

    # Compute contrastive center loss
    loss_cent = contrastive_center_loss(upsampled_embedding.view(-1, upsampled_embedding.size(1)), masks.view(-1)) * alpha

    # Total loss
    total_loss = ce_weight * loss_ce + center_weight * loss_cent

    return total_loss, loss_ce.item(), loss_cent.item()
