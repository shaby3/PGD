import torch
# from mmdet.core.bbox.iou_calculators import build_iou_calculator

eps = 1e-10

def get_back_weight(bboxes,cls_scores, bbox_preds, gt_bboxes, gt_bboxes_ignore, gt_labels):
    with torch.no_grad():
        bboxes = bboxes[:, :4]  # anchor bbox
        # bbox_preds = bbox_preds.detach()
        cls_scores = cls_scores.detach()
        device = bboxes.device

        max_cls_scores,_ = torch.sigmoid(cls_scores).max(dim=1)

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

        max_cls_scores = max_cls_scores * out_box
        max_cls_scores_value, max_cls_scores_inds = torch.topk(max_cls_scores,100)
        back_w = torch.zeros_like(out_box,device=device)
        back_w[max_cls_scores_inds] = max_cls_scores_value

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