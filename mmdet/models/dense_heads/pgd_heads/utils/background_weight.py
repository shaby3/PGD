import torch
from mmdet.core.bbox.iou_calculators import build_iou_calculator

eps = 1e-10

def get_back_weight(bboxes,cls_scores, bbox_preds, gt_bboxes, gt_bboxes_ignore, gt_labels):
    with torch.no_grad():
        bboxes = bboxes[:, :4]  # anchor bbox
        bbox_preds = bbox_preds.detach()
        cls_scores = cls_scores.detach()
        device = bboxes.device
        
        dict_iou_calculator = dict(type='BboxOverlaps2D')
        iou_calculator = build_iou_calculator(dict_iou_calculator)

        cls_score = torch.sigmoid(cls_scores[:,gt_labels])
        iou_score = iou_calculator(bbox_preds,gt_bboxes)

        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        if num_gt == 0 or num_bboxes == 0:
            return torch.zeros((num_bboxes,), dtype=cls_scores.dtype)

        x1 = bboxes[:, 0][:, None].repeat(1, num_gt)  # [n_bbox, n_gt]
        y1 = bboxes[:, 1][:, None].repeat(1, num_gt)
        x2 = bboxes[:, 2][:, None].repeat(1, num_gt)
        y2 = bboxes[:, 3][:, None].repeat(1, num_gt)

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        gx1 = gt_bboxes[:, 0][None, :].repeat(num_bboxes, 1)  # [n_bbox, n_gt]
        gy1 = gt_bboxes[:, 1][None, :].repeat(num_bboxes, 1)
        gx2 = gt_bboxes[:, 2][None, :].repeat(num_bboxes, 1)
        gy2 = gt_bboxes[:, 3][None, :].repeat(num_bboxes, 1)

        valid = ((cx - gx1) > eps) * ((cy - gy1) > eps) * ((gx2 - cx) > eps) * ((gy2 - cy) > eps)
        valid = valid.to(dtype=bboxes.dtype)

        in_box = valid.sum(dim=1)
        out_box = (in_box == 0).to(dtype=bboxes.dtype)

        alpha = 0.2
        topk = 100

        quality_score = cls_score ** (1-alpha) * iou_score ** (alpha)

        out_box = out_box.unsqueeze(dim=1)
        out_quality_score = quality_score * out_box

        max_quality_score,_ = torch.max(out_quality_score,dim=1)
        
        max_quality_scores_values, max_quality_scores_inds = torch.topk(max_quality_score,topk)
        # print('over_cls_score_inds',over_cls_score_inds.shape)

        back_w = torch.zeros_like(out_box,device=device).squeeze(dim=1)
        back_w[max_quality_scores_inds] = max_quality_scores_values

        return back_w # shape: [num_bboxes]

def get_back_weight_anchorfree(points, gt_bboxes):
    with torch.no_grad():
        num_gt, num_bboxes = gt_bboxes.size(0), points.size(0)

        cy = points[:, 0].reshape(-1, 1).repeat(1, num_gt)
        cx = points[:, 1].reshape(-1, 1).repeat(1, num_gt)

        gx1 = gt_bboxes[:, 0][None, :].repeat(num_bboxes, 1)  # [n_bbox, n_gt]
        gy1 = gt_bboxes[:, 1][None, :].repeat(num_bboxes, 1)
        gx2 = gt_bboxes[:, 2][None, :].repeat(num_bboxes, 1)
        gy2 = gt_bboxes[:, 3][None, :].repeat(num_bboxes, 1)

        valid = ((cx - gx1) > eps) * ((cy - gy1) > eps) * ((gx2 - cx) > eps) * ((gy2 - cy) > eps)
        valid = valid.to(dtype=gt_bboxes.dtype)

        in_box = valid.sum(dim=1)
        back_w = (in_box == 0).to(dtype=gt_bboxes.dtype)

        return back_w