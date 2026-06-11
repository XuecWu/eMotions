import os
import datetime
import shutil
import math
from transforms.spatial import Preprocessing
from functools import partial


def local2global_path(opt):
    if opt.root_path != '':
        opt.video_path      = os.path.join(opt.root_path, opt.video_path)
        opt.audio_path      = os.path.join(opt.root_path, opt.audio_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)


        if opt.debug:
            opt.result_path = "debug"
        opt.result_path = os.path.join(opt.root_path, opt.result_path)


        if opt.expr_name == '':
            now             = datetime.datetime.now()
            now             = now.strftime('result_%Y%m%d_%H%M%S')
            opt.result_path = os.path.join(opt.result_path, now)

        else:
            opt.result_path = os.path.join(opt.result_path, opt.expr_name)

            if os.path.exists(opt.result_path):


                if getattr(opt, 'mode', 'train') == 'train':
                    shutil.rmtree(opt.result_path)
                    os.mkdir(opt.result_path)
                else:
                    os.makedirs(opt.result_path, exist_ok=True)
            else:
                os.mkdir(opt.result_path)


        opt.log_path  = os.path.join(opt.result_path, "tensorboard")
        opt.ckpt_path = os.path.join(opt.result_path, "checkpoints")


        if not os.path.exists(opt.log_path):
            os.makedirs(opt.log_path)
        if not os.path.exists(opt.ckpt_path):
            os.mkdir(opt.ckpt_path)
    else:
        raise Exception


def get_spatial_transform_train(opt):
    return Preprocessing(size=opt.sample_size, is_aug=True, center=False)

def get_spatial_transform_val(opt):
    return Preprocessing(size=opt.sample_size, is_aug=False, center=False)

def get_spatial_transform_test(opt):
    return Preprocessing(size=opt.sample_size, is_aug=False, center=True)


def process_data_item(opt, data_item):
    visual, target, audio, visualization_item = data_item

    assert visual.size(0) == audio.size(0)
    batch  = visual.size(0)

    return visual, target, audio, visualization_item, batch


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):

    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter)))
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func                = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    elif lr_decay_type == "step":
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func        = partial(step_lr, lr, decay_rate, step_size)
    else:
        raise TypeError('lr_decay_type must be cos or step')

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def calculate_accuracy(outputs, targets):
    batch_size         = targets.size(0)
    values, indices    = outputs.topk(k=1, dim=1, largest=True)
    pred               = indices
    pred               = pred.t()
    correct            = pred.eq(targets.view(1, -1))
    n_correct_elements = correct.float()
    n_correct_elements = n_correct_elements.sum()
    n_correct_elements = n_correct_elements.item()

    return n_correct_elements / batch_size
