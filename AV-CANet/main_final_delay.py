\
\
\


from opts import parse_opts
from core.model import generate_model
from core.loss import get_loss
from core.utils import local2global_path, process_data_item, get_lr_scheduler, set_optimizer_lr, AverageMeter, calculate_accuracy,\
    get_spatial_transform_train, get_spatial_transform_val
from core.dataset import get_training_set, get_validation_set, get_test_set, get_data_loader
from transforms.temporal import TSN
from transforms.target import ClassLabel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix
from utils.plot_cm import plot_confusion_matrix
from tqdm import tqdm
import os
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

def train_epoch(epoch, model, device, data_loader, criterion, optimizer, opt, writer):
\
\
\

    model.train()
    print("#-----------------------------#")
    print("#-----------------------------#")
    print('Training at epoch {}'.format(epoch))


    _loss     = 0
    _loss_a   = 0
    _loss_v   = 0
    _loss_1   = 0

    batch_time  = AverageMeter()
    data_time   = AverageMeter()
    end_time    = time.time()

    _loss_1            = AverageMeter()
    accuracies_out     = AverageMeter()
    accuracies_a_out   = AverageMeter()
    accuracies_v_out   = AverageMeter()
    accumulation_steps = 4


    for step, data_item in enumerate(data_loader):
        visual, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)

        audio  = audio.to(device)
        visual = visual.to(device)
        target = target.to(device)

        if step % accumulation_steps == 0:
            optimizer.zero_grad()


        a, v, out = model(visual, audio)


        if opt.fusion_method == 'sum':
            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) + model.module.fusion_module.fc_y.bias)
            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) + model.module.fusion_module.fc_x.bias)
        else:
            weight_size = model.module.fusion_module.fc_out.weight.size(1)
            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1)) + model.module.fusion_module.fc_out.bias / 2)
            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1)) + model.module.fusion_module.fc_out.bias / 2)


        loss   = criterion(out, target)
        loss_v = criterion(out_v, target)
        loss_a = criterion(out_a, target)
        loss  += (loss_v + loss_a)
        loss   = loss / accumulation_steps
        loss.backward()

        acc_out = calculate_accuracy(out, target)
        acc_a   = calculate_accuracy(out_a,target)
        acc_v   = calculate_accuracy(out_v, target)

        accuracies_out.update(acc_out, batch_size)
        accuracies_a_out.update(acc_a, batch_size)
        accuracies_v_out.update(acc_v, batch_size)

        iteration = (epoch - 1) * len(data_loader) + (step + 1)


        if opt.modulation == 'Normal':
            pass
        else:
            pass


        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


        _loss_1.update(loss.item(), batch_size)
        _loss    += loss.item()
        _loss_a  += loss_a.item()
        _loss_v  += loss_v.item()

        batch_time.update(time.time() - end_time)
        end_time  = time.time()

        writer.add_scalar('train/batch/total_loss', _loss_1.val, iteration)
        writer.add_scalar('train/batch/total_acc', accuracies_out.val, iteration)

    writer.add_scalar('train/epoch/total_loss', _loss / len(data_loader), epoch)
    writer.add_scalar('train/epoch/total_acc', accuracies_out.avg, epoch)

    writer.add_scalar('train/epoch/loss_audio', _loss_a / len(data_loader), epoch)
    writer.add_scalar('train/epoch/loss_visual', _loss_v / len(data_loader), epoch)

    writer.add_scalar('train/epoch/acc_audio', accuracies_a_out.avg, epoch)
    writer.add_scalar('train/epoch/acc_visual', accuracies_v_out.avg, epoch)

    print("Total acc in Train: {:.4f}".format(accuracies_out.avg))
    print("Audio acc in Train: {:.4f}".format(accuracies_a_out.avg))
    print("Visual acc in Train: {:.4f}".format(accuracies_v_out.avg))
    print("Train Epoch Time: {:.2f}min".format(batch_time.avg * len(data_loader) / 60))


    return _loss / len(data_loader), _loss_a / len(data_loader), _loss_v / len(data_loader)


def val_epoch(epoch, model, device, data_loader, criterion, opt, writer, save_cm_path=None):
\
\
\

    softmax    = nn.Softmax(dim=1)
    n_classes  = opt.n_classes

    losses_meter = AverageMeter()
    batch_time   = AverageMeter()
    data_time    = AverageMeter()
    end_time     = time.time()

    all_targets     = []
    all_predictions = []

    class_names = {
        "Excitation": 0,
        "Fear"      : 1,
        "Neutral"   : 2,
        "Relaxation": 3,
        "Sadness"   : 4,
        "Tension"   : 5
    }


    with torch.no_grad():
        model.eval()


        num       = [0.0 for _ in range(n_classes)]
        acc       = [0.0 for _ in range(n_classes)]
        acc_a     = [0.0 for _ in range(n_classes)]
        acc_v     = [0.0 for _ in range(n_classes)]
        class_acc = [0.0 for _ in range(n_classes)]


        for step, data_item in enumerate(data_loader):
            visual, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)
            data_time.update(time.time() - end_time)

            audio  = audio.to(device)
            visual = visual.to(device)
            target = target.to(device)

            a, v, out = model(visual, audio)

            if opt.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) + model.module.fusion_module.fc_y.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) + model.module.fusion_module.fc_x.bias / 2)
            else:
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) + model.module.fusion_module.fc_out.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) + model.module.fusion_module.fc_out.bias / 2)

            losses     = criterion(out, target)
            losses_meter.update(losses.item(), batch_size)


            prediction = softmax(out)
            pred_v     = softmax(out_v)
            pred_a     = softmax(out_a)


            for i in range(visual.shape[0]):
                ma = np.argmax(prediction[i].cpu().data.numpy())
                v  = np.argmax(pred_v[i].cpu().data.numpy())
                a  = np.argmax(pred_a[i].cpu().data.numpy())


                all_targets.append(target[i].cpu().data.numpy())
                all_predictions.append([ma])

                num[target[i]] += 1.0


                if np.asarray(target[i].cpu()) == ma:
                    acc[target[i]]   += 1.0
                if np.asarray(target[i].cpu()) == v:
                    acc_v[target[i]] += 1.0
                if np.asarray(target[i].cpu()) == a:
                    acc_a[target[i]] += 1.0

            weighted_f1 = f1_score(all_targets, all_predictions, average='weighted')
            cm          = confusion_matrix(y_true=all_targets, y_pred=all_predictions, normalize='true')


            cm_uar_war  = confusion_matrix(y_true=all_targets, y_pred=all_predictions)


            batch_time.update(time.time() - end_time)
            end_time = time.time()

        recalls     = [cm_uar_war[i, i].item() / cm_uar_war[i, :].sum().item() if cm_uar_war[i, :].sum().item() != 0 else 0.0 for i in range(n_classes)]
        uar         = sum(recalls) / n_classes
        war         = sum(recalls[i] * cm_uar_war[i, :].sum().item() / cm_uar_war.sum().item() for i in range(n_classes))

        print("#-----------------------------------#")
        print('The UAR is: {:.4f}.'.format(uar * 100))
        print('The WAR is: {:.4f}.'.format(war * 100))


        print("#-----------------------------#")
        for class_name, class_index in class_names.items():
            if num[class_index] > 0:
                class_acc[class_index] = acc[class_index] / num[class_index]
            print(f'Class {class_name} Accuracy: {class_acc[class_index]:.4f}')


    if save_cm_path is not None:
        print(cm)
        print('The Confusion Matrix has been saved!')

    print("#-----------------------------#")
    print("Val Epoch Time: {:.2f}min".format(batch_time.avg * len(data_loader) / 60))
    print("Total loss in Val: {:.4f}".format(losses_meter.avg))
    print("Weighted Average F1-Score: {:.4f}".format(weighted_f1))

    writer.add_scalar('val/epoch/total_loss', losses_meter.avg, epoch)
    writer.add_scalar('val/epoch/total_acc', sum(acc) / sum(num), epoch)
    writer.add_scalar('val/epoch/total_acc_audio', sum(acc_a) / sum(num), epoch)
    writer.add_scalar('val/epoch/total_acc_visual', sum(acc_v) / sum(num), epoch)
    writer.add_scalar('val/epoch/weighted_f1_score', weighted_f1, epoch)

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def load_checkpoint_for_inference(model, checkpoint_path, device):
\
\

    if checkpoint_path == '' or checkpoint_path is None:
        raise ValueError('Please provide --resume_path for test/inference mode.')
    if os.path.isdir(checkpoint_path):
        raise IsADirectoryError(
            '{} is a directory. Please set --resume_path to a concrete .pth file.'.format(checkpoint_path)
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    loaded_dict = torch.load(checkpoint_path, map_location=device)
    state_dict  = loaded_dict['model'] if isinstance(loaded_dict, dict) and 'model' in loaded_dict else loaded_dict

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:


        model_is_dp = isinstance(model, torch.nn.DataParallel)
        has_module  = all(k.startswith('module.') for k in state_dict.keys())
        if model_is_dp and not has_module:
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif (not model_is_dp) and has_module:
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        else:
            raise
        model.load_state_dict(state_dict, strict=True)

    return loaded_dict


def main():
\
\

    opt               = parse_opts()
    print(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))


    seed    = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("#=================***==============#")
    print(f"The value of seed is {seed}")

    local2global_path(opt)


    if opt.mode == 'test':
        opt.video_pretrained = None
        opt.audio_pretrained = False


    device            = torch.device('cuda')
    model, parameters = generate_model(opt)
    model.to(device)
    model   = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()

    criterion         = get_loss(opt)
    criterion         = criterion.cuda()
    optimizer         = optim.AdamW(filter(lambda p: p.requires_grad, parameters), lr=opt.learning_rate, betas=(0.9, 0.999), weight_decay=0.02)
    lr_scheduler_func = get_lr_scheduler("cos", opt.learning_rate, opt.learning_rate*0.01, opt.n_epochs)
    writer            = SummaryWriter(logdir=opt.log_path)


    spatial_transform  = get_spatial_transform_train(opt)
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform   = ClassLabel()

    train_dataset      = get_training_set(opt, spatial_transform, temporal_transform, target_transform)
    train_dataloader   = get_data_loader(opt, train_dataset, shuffle=True)


    spatial_transform   = get_spatial_transform_val(opt)
    temporal_transform  = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform    = ClassLabel()

    val_dataset         = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
    val_dataloader      = get_data_loader(opt, val_dataset , shuffle=False)


    if opt.mode=='train':
        best_acc = 0.0
        for epoch in range(1, opt.n_epochs + 1):


            set_optimizer_lr(optimizer, lr_scheduler_func, opt.n_epochs)
            print("#========================================#")
            for param_group in optimizer.param_groups:
                print(f"The lr of ep {epoch} is: {param_group['lr']}")

            batch_loss, batch_loss_a, batch_loss_v = train_epoch(epoch, model, device, train_dataloader, criterion, optimizer, opt, writer)
            acc, acc_a, acc_v                      = val_epoch(epoch, model, device, val_dataloader, criterion, opt, writer, save_cm_path=os.path.join(opt.result_path, 'confusion_matrix'))

            if not os.path.exists(opt.ckpt_path):
                os.mkdir(opt.ckpt_path)

            if acc > best_acc:
                best_acc = float(acc)


                model_name = 'best_model_of_epoch_{}_acc_{}.pth'.format(epoch, acc)
                saved_dict = {'saved_epoch': epoch,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict()}

                save_dir   = os.path.join(opt.ckpt_path, model_name)
                torch.save(saved_dict, save_dir)
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss in Train: {:.4f}, Acc in Val: {:.4f}".format(batch_loss, acc))
                print("Audio Loss in Train: {:.4f}, Visual Loss in Train: {:.4f}".format(batch_loss_a, batch_loss_v))
                print("Audio Acc in Val: {:.4f}, Visual Acc in Val: {:.4f} ".format(acc_a, acc_v))

            else:
                print("Loss in Train: {:.4f}, Acc in Val: {:.4f}, Best Acc in Val: {:.4f}".format(batch_loss, acc, best_acc))
                print("Audio Loss in Train: {:.4f}, Visual Loss in Train: {:.4f}".format(batch_loss_a, batch_loss_v))
                print("Audio Acc in Val: {:.4f}, Visual Acc in Val: {:.4f} ".format(acc_a, acc_v))
        writer.close()

    elif opt.mode=='test':


        epoch       = 1
        loaded_dict = load_checkpoint_for_inference(model, opt.resume_path, device)
        print('Trained model loaded from {}!'.format(opt.resume_path))
        if isinstance(loaded_dict, dict):
            print('Checkpoint epoch: {}, checkpoint acc: {}'.format(
                loaded_dict.get('saved_epoch', 'N/A'), loaded_dict.get('acc', 'N/A')
            ))
        acc, acc_a, acc_v = val_epoch(epoch, model, device, val_dataloader, criterion, opt, writer)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))
    else:
        pass

if __name__ == "__main__":
    main()
