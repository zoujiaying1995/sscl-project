import torch
from torch import nn
# from torch.nn.modules.loss import _Loss
from sklearn.metrics import f1_score
from torch.nn import functional as F


class simCSELoss(nn.Module):
    def __init__(self, opt):
        super(simCSELoss, self).__init__()
        self.opt = opt
        self.temperature = opt.temperature

    def forward(self, y_pred):
        idxs = torch.arange(0, y_pred.shape[0], device='cuda')
        y_true = idxs + 1 - idxs % 2 * 2
        similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)

        similarities = similarities - torch.eye(y_pred.shape[0], device=self.opt.device) * 1e12

        similarities = similarities / self.temperature

        loss = F.cross_entropy(similarities, y_true)
        return torch.mean(loss)


class unitedCLLoss(nn.Module):
    def __init__(self, opt, contrast_mode='all'):
        super(unitedCLLoss, self).__init__()
        self.opt = opt
        self.temperature = opt.temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels, mask=None):
        """
            Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
            It also supports the unsupervised contrastive loss in SimCLR
        """
        """ Compute loss for model. If both `labels` and `mask` are None,
            it degenerates to SimCLR unsupervised loss:
            https://arxiv.org/pdf/2002.05709.pdf
            Args:
                features: hidden vector of shape [bsz, n_views, ...].
                labels: ground truth of shape [bsz].
                mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                    has the same class as sample i. Can be asymmetric.
            Returns:
                A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                labels = torch.cat([labels, labels], dim=0)

            mask = torch.eq(labels, labels.T).float().add(0.0000001).to(
                device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask) - mask) * logits_mask

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)

        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))

        return loss
