import torch
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from .builder import DISTILL_WEIGHT

eps = 1e-10

@DISTILL_WEIGHT.register_module()
class PGWAnchorModule(torch.nn.Module):
    def __init__(self,
                 alpha,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1,
                 low_bound=0.,):

        super(PGWAnchorModule, self).__init__()
        self.alpha = alpha
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr
        self.low_bound = low_bound

    @torch.no_grad()
    def assign(self,
               bboxes,
               cls_scores,
               bbox_preds,
               gt_bboxes,
               bbox_levels,
               positive_inds,
               gt_bboxes_ignore=None,
               gt_labels=None):

        bboxes = bboxes[:, :4]  # anchor bbox
        bbox_preds = bbox_preds.detach()
        cls_scores = cls_scores.detach()
        device = bboxes.device

        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        if num_gt == 0 or num_bboxes == 0:
            return torch.zeros((num_bboxes,), dtype=cls_scores.dtype)

        overlaps = self.iou_calculator(bbox_preds, gt_bboxes)
        cls_cost = torch.sigmoid(cls_scores[:, gt_labels])  # [num_bbox, num_gt]
        assert cls_cost.shape == overlaps.shape

        overlaps = cls_cost ** (1 - self.alpha) * overlaps ** self.alpha

        quality_score = torch.max(overlaps,dim=1)[0] # shape: [num_bbox]
        
        pos = torch.zeros_like(quality_score,device=device)
        pos[positive_inds] = 1

        assigned_gt_inds = overlaps.new_full((num_bboxes,), 0, dtype=torch.long)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            assigned_gt_inds[ignore_idxs] = -1

        quality_score = quality_score * pos

        quality_score[assigned_gt_inds == -1] = 0.
        quality_score[quality_score < self.low_bound] = 0.

        return quality_score

    def forward(self, **kwargs):
        pass


def mle_2d_gaussian_2(sampled_data):
    # sampled_data [N, M, 2]
    data = sampled_data + (torch.rand(sampled_data.size(), device=sampled_data.device) - 0.5) * 0.1  # hack: add noise to avoid zero determint
    miu = data.mean(dim=1, keepdim=True) #[N, 1, 2]
    diff = (data - miu)[:, :, :, None]
    sigma = torch.matmul(diff, diff.transpose(2, 3)).mean(dim=1)  # [N, 2, 2]
    deter = sigma[:, 0, 0] * sigma[:, 1, 1] - sigma[:, 0, 1] * sigma[:, 1, 0]  # [N]

    inverse = torch.zeros_like(sigma)
    inverse[:, 0, 0] = sigma[:, 1, 1]
    inverse[:, 0, 1] = -1. * sigma[:, 0, 1]
    inverse[:, 1, 0] = -1. * sigma[:, 1, 0]
    inverse[:, 1, 1] = sigma[:, 0, 0]
    inverse /= (deter[:,None,None]+1e-10)
    return miu, sigma, inverse, deter










