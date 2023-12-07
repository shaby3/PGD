import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import constant_init, kaiming_init
from mmdet.distillation.builder import DISTILL_LOSSES
from mmcv.runner import force_fp32

eps = 1e-10

@DISTILL_LOSSES.register_module()
class PGDRegLoss(nn.Module):
    def __init__(self,
                 name,
                 loss_weight,
                 alpha,
                 beta,
                 **kwargs,
                 ):
        super(PGDRegLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta

    @force_fp32(apply_to=('preds_S', 'preds_T'))
    def forward(self,
                preds_S,
                preds_T,
                mask_fg,
                mask_bg,
                **kwargs):
        assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'

        mask_bg = mask_bg / (mask_bg != 0).to(dtype=preds_S.dtype).sum(dim=(2, 3), keepdims=True).clamp(min=eps) # shape: [bs,1,h,w] # bs,1,h,w / bs,1,1,1
        mask_fg = mask_fg / (mask_fg != 0).to(dtype=preds_S.dtype).sum(dim=(2, 3), keepdims=True).clamp(min=eps) # shape: [bs,1,h,w] # bs,1,h,w / bs,1,1,1

        fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T,
                                             mask_fg, mask_bg,
                                            )

        loss = (self.loss_weight * self.alpha * fg_loss)# + (self.loss_weight * self.beta * bg_loss)

        return loss


    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg):
        loss_mse = nn.MSELoss(reduction='sum')


        fg_fea_t = torch.mul(preds_T, torch.sqrt(Mask_fg)) # shape: [bs,bs,c,h,w]
        bg_fea_t = torch.mul(preds_T, torch.sqrt(Mask_bg)) # shape: [bs,bs,c,h,w]

        fg_fea_s = torch.mul(preds_S, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(preds_S, torch.sqrt(Mask_bg))

        # batch size로 나눠주기
        fg_loss = loss_mse(fg_fea_s, fg_fea_t) / len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t) / len(Mask_bg)

        return fg_loss, bg_loss

