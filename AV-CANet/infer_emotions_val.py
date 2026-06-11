\
\
\


import argparse
import csv
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

from core.dataset import get_data_loader, get_validation_set
from core.model import generate_model
from core.utils import get_spatial_transform_test, get_spatial_transform_val, process_data_item
from transforms.target import ClassLabel
from transforms.temporal import TSN


DEFAULT_CKPT = ""


def parse_args():
\
\

    parser = argparse.ArgumentParser(description="Run AV-CANet inference on eMotions validation set.")


    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--video_path", type=str, default="eMotions_all_processed/eMotions_imgs")
    parser.add_argument("--audio_path", type=str, default="eMotions_all_processed/eMotions_mp3")
    parser.add_argument("--annotation_path", type=str, default="annotations/eMotions6_01_all.json")
    parser.add_argument("--checkpoint_path", "--resume_path", dest="checkpoint_path", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--save_dir", type=str, default="results/inference_eMotions6_01_all_val")


    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sample_size", type=int, default=112)
    parser.add_argument("--snippet_duration", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--audio_embed_size", type=int, default=512)
    parser.add_argument("--audio_n_segments", type=int, default=16)
    parser.add_argument("--fusion_method", type=str, default="concat", choices=["sum", "concat", "gated", "film"])
    parser.add_argument("--dataset", type=str, default="ve8")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1)


    parser.add_argument("--video_pretrained", type=str, default="")
    parser.add_argument("--audio_pretrained", action="store_true", default=False)


    parser.add_argument("--reproduce_train_val_transform", action="store_true", default=False)

    return parser.parse_args()


def make_abs(root_path, path):
    if path is None or path == "":
        return path
    return path if os.path.isabs(path) else os.path.join(root_path, path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_checkpoint(model, checkpoint_path, device):
\
\

    if os.path.isdir(checkpoint_path):
        raise IsADirectoryError(f"checkpoint_path points to a directory: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint


    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[Warning] Missing keys when loading checkpoint:")
        for k in missing[:20]:
            print("  ", k)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    if unexpected:
        print("[Warning] Unexpected keys when loading checkpoint:")
        for k in unexpected[:20]:
            print("  ", k)
        if len(unexpected) > 20:
            print(f"  ... and {len(unexpected) - 20} more")

    return checkpoint


def get_video_ids(visualization_item):
    """Normalize default_collate output for visualization_item=[video_id]."""
    if isinstance(visualization_item, (list, tuple)):
        if len(visualization_item) == 1 and isinstance(visualization_item[0], (list, tuple)):
            return list(visualization_item[0])
        return [x[0] if isinstance(x, (list, tuple)) else x for x in visualization_item]
    return [str(visualization_item)]


def modal_logits(model, a, v, opt):
\
\

    if opt.fusion_method == "sum":
        out_v = torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) + model.fusion_module.fc_y.bias
        out_a = torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) + model.fusion_module.fc_x.bias
    else:
        weight_size = model.fusion_module.fc_out.weight.size(1)
        half = weight_size // 2
        out_v = torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, half:], 0, 1)) + model.fusion_module.fc_out.bias / 2
        out_a = torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, :half], 0, 1)) + model.fusion_module.fc_out.bias / 2
    return out_a, out_v


def main():
\
\

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("This project currently requires CUDA because VideoSwin/AVNet call .cuda() internally.")
    device = torch.device("cuda")

    opt = SimpleNamespace(**vars(args))
    opt.video_path = make_abs(opt.root_path, opt.video_path)
    opt.audio_path = make_abs(opt.root_path, opt.audio_path)
    opt.annotation_path = make_abs(opt.root_path, opt.annotation_path)
    opt.save_dir = make_abs(opt.root_path, opt.save_dir)
    opt.dl = False
    opt.use_cuda = True
    opt.mode = "test"


    opt.video_pretrained = opt.video_pretrained if opt.video_pretrained else None

    os.makedirs(opt.save_dir, exist_ok=True)

    print("#================ Inference Configuration ================#")
    print(f"Annotation : {opt.annotation_path}")
    print(f"Video root : {opt.video_path}")
    print(f"Audio root : {opt.audio_path}")
    print(f"Checkpoint : {opt.checkpoint_path}")
    print(f"Save dir   : {opt.save_dir}")

    model, _ = generate_model(opt)
    model = model.to(device)
    checkpoint = load_checkpoint(model, opt.checkpoint_path, device)
    model.eval()

    if isinstance(checkpoint, dict):
        print(f"Loaded checkpoint epoch: {checkpoint.get('saved_epoch', 'N/A')}")
        print(f"Loaded checkpoint acc  : {checkpoint.get('acc', 'N/A')}")

    if opt.reproduce_train_val_transform:
        spatial_transform = get_spatial_transform_val(opt)
        temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
        print("Using original random validation transform.")
    else:
        spatial_transform = get_spatial_transform_test(opt)
        temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=True)
        print("Using deterministic center-crop validation transform.")

    target_transform = ClassLabel()
    val_dataset = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
    val_loader = get_data_loader(opt, val_dataset, shuffle=False)

    idx_to_class = val_dataset.class_names
    class_names = [idx_to_class[i] for i in range(opt.n_classes)]
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()

    rows = []
    all_targets, all_preds = [], []
    all_preds_a, all_preds_v = [], []
    total_loss, total_count = 0.0, 0


    with torch.no_grad():
        for data_item in tqdm(val_loader, desc="Infer validation"):
            visual, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)
            visual = visual.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            a, v, out = model(visual, audio)
            out_a, out_v = modal_logits(model, a, v, opt)

            loss = criterion(out, target)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            prob = softmax(out)
            prob_a = softmax(out_a)
            prob_v = softmax(out_v)

            pred = prob.argmax(dim=1)
            pred_a = prob_a.argmax(dim=1)
            pred_v = prob_v.argmax(dim=1)

            video_ids = get_video_ids(visualization_item)
            target_cpu = target.cpu().numpy().tolist()
            pred_cpu = pred.cpu().numpy().tolist()
            pred_a_cpu = pred_a.cpu().numpy().tolist()
            pred_v_cpu = pred_v.cpu().numpy().tolist()
            prob_cpu = prob.cpu().numpy()

            all_targets.extend(target_cpu)
            all_preds.extend(pred_cpu)
            all_preds_a.extend(pred_a_cpu)
            all_preds_v.extend(pred_v_cpu)

            for i, vid in enumerate(video_ids):
                row = {
                    "video_id": vid,
                    "target_id": target_cpu[i],
                    "target_label": class_names[target_cpu[i]],
                    "pred_id": pred_cpu[i],
                    "pred_label": class_names[pred_cpu[i]],
                    "pred_audio_id": pred_a_cpu[i],
                    "pred_audio_label": class_names[pred_a_cpu[i]],
                    "pred_visual_id": pred_v_cpu[i],
                    "pred_visual_label": class_names[pred_v_cpu[i]],
                    "correct": int(target_cpu[i] == pred_cpu[i]),
                }
                for c, name in enumerate(class_names):
                    row[f"prob_{name}"] = float(prob_cpu[i, c])
                rows.append(row)

    acc = accuracy_score(all_targets, all_preds)
    acc_a = accuracy_score(all_targets, all_preds_a)
    acc_v = accuracy_score(all_targets, all_preds_v)
    wa_f1 = f1_score(all_targets, all_preds, average="weighted")

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(opt.n_classes)))
    cm_norm = confusion_matrix(all_targets, all_preds, labels=list(range(opt.n_classes)), normalize="true")
    recalls = [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0 for i in range(opt.n_classes)]
    uar = float(np.mean(recalls))
    war = float(acc)
    class_acc = {class_names[i]: float(recalls[i]) for i in range(opt.n_classes)}

    pred_csv = os.path.join(opt.save_dir, "validation_predictions.csv")
    metrics_json = os.path.join(opt.save_dir, "validation_metrics.json")
    cm_csv = os.path.join(opt.save_dir, "validation_confusion_matrix.csv")
    cm_norm_csv = os.path.join(opt.save_dir, "validation_confusion_matrix_normalized.csv")

    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    np.savetxt(cm_csv, cm, fmt="%d", delimiter=",")
    np.savetxt(cm_norm_csv, cm_norm, fmt="%.8f", delimiter=",")

    metrics = {
        "num_samples": len(all_targets),
        "acc": float(acc),
        "audio_acc": float(acc_a),
        "visual_acc": float(acc_v),
        "wa_f1": float(wa_f1),
        "uar": uar,
        "war": war,
        "class_acc": class_acc,
        "class_names": class_names,
        "checkpoint_path": opt.checkpoint_path,
        "annotation_path": opt.annotation_path,
        "transform": "original_random_val" if opt.reproduce_train_val_transform else "deterministic_center_crop",
    }

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("#================ Inference Results ================#")
    print(f"Samples     : {len(all_targets)}")
    print(f"ACC/WAR     : {acc:.6f}")
    print(f"WA-F1       : {wa_f1:.6f}")
    print(f"UAR         : {uar:.6f}")
    print(f"Audio ACC   : {acc_a:.6f}")
    print(f"Visual ACC  : {acc_v:.6f}")

    for name, value in class_acc.items():
        print(f"{name:10s} : {value:.6f}")
    print("#================ Saved Files ================#")
    print(f"Predictions : {pred_csv}")
    print(f"Metrics     : {metrics_json}")
    print(f"CM          : {cm_csv}")
    print(f"CM Norm     : {cm_norm_csv}")


if __name__ == "__main__":
    main()
