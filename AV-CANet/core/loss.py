\
\
\
\
\
\
\
\


import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch import Tensor


class PCCE_Loss(nn.Module):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

    def __init__(self, lambda_POS=0.0, lambda_NEU=0.0, lambda_NEG=0.0):
        super(PCCE_Loss, self).__init__()


        self.POSITIVE   = {0, 3}
        self.NEUTRAL    = {2}
        self.NEGATIVE   = {1, 4, 5}

        self.lambda_POSITIVE = lambda_POS
        self.lambda_NEUTRAL  = lambda_NEU
        self.lambda_NEGATIVE = lambda_NEG

        self.f0       = nn.CrossEntropyLoss(reduction='none')

    def forward(self, y_pred: Tensor, y: Tensor):
\
\
\


        batch_size = y_pred.size(0)
        weight     = [1] * batch_size


        out        = self.f0(y_pred, y)


        _, y_pred_label    = F.softmax(y_pred, dim=1).topk(k=1, dim=1)
        y_pred_label       = y_pred_label.squeeze(dim=1)


        y_numpy            = y.cpu().numpy()
        y_pred_label_numpy = y_pred_label.cpu().numpy()


        for i, y_numpy_i, y_pred_label_numpy_i in zip(range(batch_size), y_numpy, y_pred_label_numpy):


            if (y_numpy_i in self.POSITIVE and y_pred_label_numpy_i in self.NEGATIVE) or\
                    (y_numpy_i in self.POSITIVE and y_pred_label_numpy_i in self.NEUTRAL):
                weight[i] += self.lambda_POSITIVE


            if (y_numpy_i in self.NEUTRAL and y_pred_label_numpy_i in self.POSITIVE) or\
                    (y_numpy_i in self.NEUTRAL and y_pred_label_numpy_i in self.NEGATIVE):
                weight[i] += self.lambda_NEUTRAL


            if (y_numpy_i in self.NEGATIVE and y_pred_label_numpy_i in self.POSITIVE) or\
                    (y_numpy_i in self.NEGATIVE and y_pred_label_numpy_i in self.NEUTRAL):
                weight[i] += self.lambda_NEGATIVE


        weight_tensor = torch.from_numpy(np.array(weight)).cuda()
        out           = out.mul(weight_tensor)
        out           = torch.mean(out)

        return out


def get_loss(opt):
\
\
\
\


    if opt.loss_func == 'ce':
        return nn.CrossEntropyLoss()


    elif opt.loss_func == 'PCCE_Loss':
        return PCCE_Loss(lambda_POS = opt.lambda_POS, lambda_NEU = opt.lambda_NEU, lambda_NEG= opt.lambda_NEG)


    else:
        raise Exception
