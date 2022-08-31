# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model  targets[n,6(b,cls,x,y,w,h)]
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        #lcls[3][n,1(cls)]
        #tbox[3][n,4(xywh)]
        #indices[3][4(b,a,gj,gi)][n]
        #anchors[3][n,2(aw,ah)]
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            #pi[b,a,gj,gi,4(rect)+1(obj)+cls]
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            #[n]
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            #tobj[b,a,gj,gi]
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                #[b,a,gj,gi,1(obj)+4(rect[xywh])+cls]-->ps[n,4(rect[xywh])+1(obj)+cls]
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                # pxy[n,2]
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                # pwh[n,2]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                #pbox[n,4]
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                # iou[n]
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                # score_iou[n]
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio
                # 所有目标对应的网格anchors都给1，其他都在初始化时刷0

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # ps[:, 5:]的shape是[n, cls]  cn==0  全部填0
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    # t[n, cls]
                    # tcls[i]的shape是[n]  cp==1
                    # range(n)是一个shape为[n]的数组，下标从0~n-1
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE MSE
                    #det1_cls = t-ps[:, 5:]
                    #lcls2 += self.BCEcls(ps2[:, 5:], det1_cls)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)#MseLoss
            lobj += obji * self.balance[i]  # obj loss
            #det1 = tobj - pi[..., 4]
            #lobj2 = self.MseLoss(pi2[..., 4], det1)
            if self.autobalance: #多个不同尺度直接的学习因子平衡
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # p[list][b,a,gy,gx,4+1+cls]
        # targets[nt,6=(1(b)+1(cls)+4(box)]
        # targets = targets[max(targets[:, 4],targets[:, 5]).max(1)[0] > 0.01]
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        # gain[7] = {1,1,1,1,1,1,1}
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # ai[na,nt]
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        # targets[na,nt,7=(1(b)+1(cls)+4(box)+1(anchor))]

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
        # off[5,2]

        for i in range(self.nl):
            # anchors[3,na,2]
            anchors = self.anchors[i]
            # anchors[na,2]
            # p[i][b,a,gy,gx,4+1+cls]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # gain[7=(1(b)+1(cls)+4(box)+1(anchor))] = {1,1,gx,gy,gx,gy,1}

            # Match targets to anchors
            t = targets * gain
            # t[na,nt,7=(1(b)+1(cls)+4(box)+1(anchor))]
            if nt:
                # Matches
                # t[:, :, 4:6]的shape是    t      [na,nt,2]
                # anchors[:, None]的shape是anchors[na, ?,2]  得到?=nt
                #                                [na,nt,2]
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                # r[na,nt,2] / anchors[na,?,2] = r[na,nt,2]

                # r1 = torch.max(r, 1. / r)的shape是r1[na,nt,2]
                # r1.max(2)意思是[r和1/r]在编号(2)的维度找最大值[0]，求最大值后会把维度2去掉,shape变为[na,nt]
                # 得到r1.max(2)[0]是shape为[na,nt]的最大值
                # 得到r1.max(2)[1]是shape为[na,nt]的最大值对应的整数编号{0,1}
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # 意思是[r和1/r]在编号(2)的维度找最大值[0],求最大值后会把维度2去掉，j的shape为[na,nt]的bool数组
                # r, 1. / r两者最大值都比较小，说明接近1，说明和相应anchor比较匹配

                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # j[na, nt]
                # t[na, nt, 7 = (1(b) + 1(cls) + 4(box) + 1(anchor))]
                t = t[j]  # filter
                # t[n_match_obj,7=(1(b)+1(cls)+4(box)+1(anchor))]

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                # gxy[n_match_obj,2=(xy)]
                gxi = gain[[2, 3]] - gxy  # inverse
                # gxi[n_match_obj,2=(xy)]
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                # j,k,l,m的shape全是[n_match_obj]
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # j[5,n_match_obj] [第1行全1/目标中心x接近下限神经元j/目标中心y接近下限神经元k/目标中心x接近上限神经元l/目标中心y接近上限神经元m]
                # t.repeat((5, 1, 1))的shape是[5,n_match_obj,7 = (1(b) + 1(cls) + 4(box) + 1(anchor))]
                t = t.repeat((5, 1, 1))[j]
                # t[n_match_obj*3,7]  目标第1行全1，[目标x要么接近下限x，要么接近上限x],[目标y要么接近下限y，要么接近上限y],因此只有可能是选择3倍

                # gxy[n_match_obj, 2 = (xy)]
                # torch.zeros_like(gxy)[None]的shape是[?, n_match_obj, 2 = (xy)]
                # off[5,2]
                # off[:, None]的shape是[5, ?, 2]
                # [?1, n_match_obj, 2]和
                # [ 5,          ?2, 2]
                # 要能在一起计算，只有?1=5, ?2=n_match_obj  最终得到的shape是
                # [ 5, n_match_obj, 2]
                #得到(torch.zeros_like(gxy)[None] + off[:, None])的shape是[5, n_match_obj, 2]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                #[5, n_match_obj, 2]经过j[5,n_match_obj]过滤得到offsets[n_match_obj * 3, 2]
                #相当于把n_match_obj*5的组合全部遍历了一遍,选择其中n_match_obj*3的就近的网格偏移
            else:
                t = targets[0]
                offsets = 0

            # Define
            # t[:, :2]的shape是[3*n_match_obj,2]
            b, c = t[:, :2].long().T  # image, class
            # b[3*n_match_obj]  c[3*n_match_obj]
            # t[:, 2:4]的shape是[3*n_match_obj,2]
            gxy = t[:, 2:4]  # grid xy
            # gxy[3*n_match_obj,2]   得到3*n_match_obj个gxy网格(xy坐标复制了3倍)
            gwh = t[:, 4:6]  # grid wh
            # gwh[3*n_match_obj,2]   得到3*n_match_obj个gwh网格(wh坐标复制了3倍)
            gij = (gxy - offsets).long()
            # gij[3*n_match_obj,2]   得到3*n_match_obj个就近网格
            gi, gj = gij.T  # grid xy indices
            # gi[3*n_match_obj]  gj[3*n_match_obj]

            # Append
            a = t[:, 6].long()  # anchor indices
            # a[3*n_match_obj]
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            # gxy - gij才是网络真正要拟合的相对偏移！
            # anchors[3,2]   a[3*n_match_obj]
            # anchors[a] shape是[3*n_match_obj,2]
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
