from __future__ import print_function, division
import argparse
import os
import subprocess

DEFAULT_CLASSES = ["Excitation", "Fear", "Neutral", "Relaxation", "Sadness", "Tension"]


def class_process(dir_path, dst_dir_path, class_name):

    src_class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(src_class_path):
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_class_path):
        os.makedirs(dst_class_path)

    for file_name in os.listdir(src_class_path):

        name, ext       = os.path.splitext(file_name)
        music_file_name = name + '.mp3'
        video_file_path = os.path.join(src_class_path, file_name)
        music_file_path = os.path.join(dst_class_path, music_file_name)

        cmd = 'ffmpeg -i \"{}\" \"{}\"'.format(video_file_path, music_file_path)
        print(cmd)
        subprocess.call(cmd, shell=True)
        print('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mp3 audio from class-organized video folders.")
    parser.add_argument("--input_dir", required=True, help="Root directory of raw videos organized by class.")
    parser.add_argument("--output_dir", required=True, help="Output root directory for extracted mp3 files.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="Class names to process.")
    args = parser.parse_args()

    for class_name in args.classes:
        class_process(args.input_dir, args.output_dir, class_name)
