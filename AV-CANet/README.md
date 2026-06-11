# AV-CANet

AV-CANet is an audio-visual baseline for emotion recognition in short-form videos. The model combines a Video Swin visual branch, a ResNet-style audio branch, and cross-modal fusion modules for eMotions experiments.

## Repository Scope

Included:

- Audio-visual model implementation
- Training, validation, and inference entry points
- Dataset loader and preprocessing utilities
- Frame and audio preprocessing scripts

Not included:

- eMotions dataset files
- Annotation JSON or split txt files
- Checkpoints, pretrained weights, logs, or inference outputs

The eMotions dataset is hosted separately at:

https://huggingface.co/datasets/Conna/eMotions

## Structure

```text
.
├── main_final_delay.py       # training and test entry point
├── infer_emotions_val.py     # standalone validation/inference script
├── opts.py                   # command-line configuration
├── models/                   # audio, visual, and fusion modules
├── core/                     # losses, model factory, and utilities
├── datasets/                 # eMotions-style dataset loader
├── transforms/               # spatial, temporal, and target transforms
└── tools/                    # preprocessing helpers
```

## Installation

```bash
conda create -n avcanet python=3.8 -y
conda activate avcanet
pip install -r requirements.txt
```

Install the PyTorch/CUDA build that matches your environment. `ffmpeg` is required for preprocessing.

## Data Preparation

Download the dataset from HuggingFace and prepare a local directory layout:

```text
AV-CANet/
  eMotions_all_processed/
    eMotions_imgs/<class>/<video_id>/000001.jpg ... n_frames
    eMotions_mp3/<class>/<video_id>.mp3
  annotations/eMotions6_01_all.json
```

The annotation JSON and split files are not committed. Put them under `annotations/` locally or pass explicit paths at runtime.

## Preprocessing

Extract video frames:

```bash
python tools/video2jpg.py \
  --input_dir /path/to/eMotions/video_raw \
  --output_dir eMotions_all_processed/eMotions_imgs
```

Extract audio:

```bash
python tools/video2mp3.py \
  --input_dir /path/to/eMotions/video_raw \
  --output_dir eMotions_all_processed/eMotions_mp3
```

Generate frame count files:

```bash
python tools/n_frames.py \
  --frame_dir eMotions_all_processed/eMotions_imgs
```

Convert split files to JSON annotations:

```bash
python tools/eMotions6_json.py \
  --csv_dir /path/to/split_txt_dir \
  --split_index 1 \
  --output annotations/eMotions6_01_all.json
```

## Training

```bash
python main_final_delay.py \
  --root_path . \
  --video_path eMotions_all_processed/eMotions_imgs \
  --audio_path eMotions_all_processed/eMotions_mp3 \
  --annotation_path annotations/eMotions6_01_all.json \
  --video_pretrained pretrained/swin_tiny_patch4_window7_224_22k.pth \
  --result_path results \
  --expr_name avcanet_emotions \
  --mode train
```

## Inference

```bash
python infer_emotions_val.py \
  --root_path . \
  --video_path eMotions_all_processed/eMotions_imgs \
  --audio_path eMotions_all_processed/eMotions_mp3 \
  --annotation_path annotations/eMotions6_01_all.json \
  --checkpoint_path /path/to/checkpoint.pth \
  --save_dir results/inference_eMotions6_val
```

## Notes

- `pretrained/`, `annotations/`, `results/`, dataset folders, and checkpoints are ignored by git.
- The training code is designed for CUDA execution.
- A full AV-CANet checkpoint should contain both visual and audio branch weights for inference.
