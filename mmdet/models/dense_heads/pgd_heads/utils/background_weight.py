import torch
import torch.nn.functional as F
# from mmdet.core.bbox.iou_calculators import build_iou_calculator

eps = 1e-10

def get_back_weight(bboxes,cls_scores, bbox_preds,teacher_feat,level_anchor, gt_bboxes, gt_bboxes_ignore, gt_labels):
    with torch.no_grad():
        bboxes = bboxes[:, :4]  # anchor bbox
        # bbox_preds = bbox_preds.detach()
        cls_scores = cls_scores.detach()
        device = bboxes.device
        teacher_feat = teacher_feat.detach()
        
        value = torch.abs(teacher_feat)
        fea_map = torch.mean(value, dim=1)
        
        max_cls_scores,_ = torch.sigmoid(cls_scores[:,gt_labels]).max(dim=1)

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

        split_fea_map = torch.split(fea_map, level_anchor) 
        out_box = torch.split(out_box, level_anchor)

        out_att_list = []
        for level,fea_per_level in enumerate(split_fea_map):
            att_per_level = len(fea_per_level) * torch.softmax(fea_per_level,dim=-1)
            att_per_level = att_per_level * out_box[level]
            out_att_list.append(att_per_level)
        out_att = torch.cat(out_att_list)
        
        alpha = 0.2
        
        new_qual = max_cls_scores ** (1-alpha) * out_att ** alpha

        _, top_att_inds = torch.topk(new_qual,100)
        back_w = torch.zeros_like(out_att,device=device)
        back_w[top_att_inds] = 1

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