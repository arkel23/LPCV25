import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.loss import LabelSmoothingCrossEntropy

from .focal_loss import FocalLoss, StudentTeacherDeltaFocalLoss
from .mix import mixup_criterion
from .contrastive_loss import SupConLoss, FocallyModulatedSupConLoss


# Center Loss for Attention Regularization
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, output, targets):
        return self.l2_loss(output, targets) / output.size(0)


# Overall CAL Loss
class CALLoss(nn.Module):
    def __init__(self):
        super(CALLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.center_loss = CenterLoss()

    def forward(self, output, y):
        if isinstance(output, tuple) and len(output) == 7:
            (_, y_pred_raw, y_pred_aux, feature_matrix, feature_center_batch,
             y_pred_aug, _) = output

            y_aug = torch.cat([y, y], dim=0)
            y_aux = torch.cat([y, y_aug], dim=0)
 
            batch_loss = (self.cross_entropy_loss(y_pred_raw, y) / 3. +
                          self.cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3. +
                          self.cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3. +
                          self.center_loss(feature_matrix, feature_center_batch))

        elif isinstance(output, tuple) and len(output) == 2:
            y_pred, _ = output
            batch_loss = self.cross_entropy_loss(y_pred, y)

        else:
            batch_loss = self.cross_entropy_loss(output, y)

        return batch_loss


class OverallLoss(nn.Module):
    def __init__(self, args, kd=False):
        super(OverallLoss, self).__init__()

        self.args = args

        if args.selector == 'cal' and not kd:
            self.criterion = CALLoss()
        elif args.focal_gamma:
            self.criterion = FocalLoss(args.focal_gamma, smoothing=args.ls)
        elif args.ls:
            self.criterion = LabelSmoothingCrossEntropy(args.smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        if kd:
            self.criterion_kd = torch.nn.KLDivLoss(reduction="batchmean")
            self.kd_temp = args.temp

            if args.kd_aux_loss == 'crd':
                self.criterion_crd = True

            elif args.kd_aux_loss == 'std_focal':
                self.modulation_augs = args.modulation_augs
                self.criterion_std_focal = StudentTeacherDeltaFocalLoss(
                    args.focal_modulation, args.modulation_teacher_labels,
                    args.cont_focal_gamma, args.cont_focal_alpha, args.ls, args.smoothing
                )

            elif args.cont_loss and args.cont_focal_modulation:
                self.focal_modulation = args.cont_focal_modulation
                self.criterion_cont = FocallyModulatedSupConLoss(
                    args.supcon, args.cont_focal_detach, args.device,
                    args.cont_focal_gamma, args.cont_focal_alpha,
                    args.cont_temp, args.cont_base_temp, args.cont_norm_ind)
                print('Modulation for Contrastive Loss: ', self.focal_modulation)

            elif args.cont_loss:
                self.criterion_cont = SupConLoss(
                    args.cont_temp, args.cont_base_temp, args.cont_norm_ind)

    def forward(self, output, targets, output_t=None, y_a=None, y_b=None, lam=None):

        if hasattr(self, 'criterion_kd') and output_t is not None:

            if self.args.tgda:
                # and hasattr(self, 'criterion_cont') and :
                if isinstance(output, tuple) and len(output) == 4:
                    output, output_aug, feats, _ = output
                    output_t, output_t_aug, _, _, _, _, _ = output_t
                elif isinstance(output, tuple) and len(output) == 3:
                    output, output_aug, _ = output
                    output_t, output_t_aug, _  = output_t


                loss_kd = self.criterion_kd(
                    F.log_softmax(output / self.kd_temp, dim=1),
                    F.softmax(output_t / self.kd_temp, dim=1)
                    ) / 2 + self.criterion_kd(
                    F.log_softmax(output_aug / self.kd_temp, dim=1),
                    F.softmax(output_t_aug / self.kd_temp, dim=1)
                    ) * 2 / 2


                if hasattr(self, 'criterion_std_focal') and hasattr(self, 'modulation_augs'):
                    loss_std = ((self.criterion_std_focal(output, output_t, targets)) / 2 +
                        (self.criterion_std_focal(output_aug, output_t_aug, targets) * 2 / 2)
                    )
                elif hasattr(self, 'criterion_std_focal'):
                    loss_std = self.criterion_std_focal(output, output_t, targets)


                if hasattr(self, 'criterion_cont') and hasattr(self, 'focal_modulation') and self.focal_modulation == 'student_teacher':
                    loss_cont = self.criterion_cont(feats, output, targets, output_t)
                elif hasattr(self, 'criterion_cont') and hasattr(self, 'focal_modulation') and self.focal_modulation == 'teacher':
                    loss_cont = self.criterion_cont(feats, output_t, targets)
                elif hasattr(self, 'criterion_cont') and hasattr(self, 'focal_modulation') and self.focal_modulation == 'student':
                    loss_cont = self.criterion_cont(feats, output, targets)
                elif hasattr(self, 'criterion_cont') and self.args.supcon:
                    loss_cont = self.criterion_cont(feats, targets)
                elif hasattr(self, 'criterion_cont'):
                    loss_cont = self.criterion_cont(feats)


            else:
                if isinstance(output, tuple) and len(output) == 3:
                    output, feats, loss_crd = output
                    output_t, _, _ = output_t
                elif isinstance(output, tuple) and len(output) == 2 and hasattr(self, 'criterion_crd'):
                    output, loss_crd = output
                    output_t, _ = output_t
                elif isinstance(output, tuple) and len(output) == 2:
                    output, feats = output
                    output_t, _, _ = output_t


                loss_kd = self.criterion_kd(
                    F.log_softmax(output / self.kd_temp, dim=1),
                    F.softmax(output_t / self.kd_temp, dim=1))


                if hasattr(self, 'criterion_std_focal'):
                    loss_std = self.criterion_std_focal(output, output_t, targets)


                if hasattr(self, 'criterion_cont') and hasattr(self, 'focal_modulation') and self.focal_modulation == 'student_teacher':
                    loss_cont = self.criterion_cont(feats, output, targets, output_t)
                elif hasattr(self, 'criterion_cont') and hasattr(self, 'focal_modulation') and self.focal_modulation == 'teacher':
                    loss_cont = self.criterion_cont(feats, output_t, targets)
                elif hasattr(self, 'criterion_cont') and hasattr(self, 'focal_modulation') and self.focal_modulation == 'student':
                    loss_cont = self.criterion_cont(feats, output, targets)
                elif hasattr(self, 'criterion_cont') and self.args.supcon:
                    loss_cont = self.criterion_cont(feats, targets)
                elif hasattr(self, 'criterion_cont'):
                    loss_cont = self.criterion_cont(feats)


        if y_a is not None:
            loss = mixup_criterion(self.criterion, output, y_a, y_b, lam)
        else:
            loss = self.criterion(output, targets)


        if hasattr(self, 'criterion_kd') and output_t is not None:
            loss = self.args.loss_orig_weight * loss + self.args.loss_kd_weight * loss_kd

        if hasattr(self, 'criterion_crd') and output_t is not None:
            loss = loss + self.args.loss_kd_aux_weight * loss_crd
        elif hasattr(self, 'criterion_std_focal') and output_t is not None:
            loss = loss + self.args.loss_kd_aux_weight * loss_std
            
        if hasattr(self, 'criterion_cont') and output_t is not None:
            loss = loss + self.args.loss_cont_weight * loss_cont

        if self.args.selector == 'cal' and isinstance(output, tuple) and len(output) == 7:
            output, _, _, _, _, _, _ = output
        elif self.args.selector == 'cal' and isinstance(output, tuple) and len(output) == 2:
            output, _ = output


        assert math.isfinite(loss), f'Loss is not finite: {loss}, stopping training'

        return output, loss
