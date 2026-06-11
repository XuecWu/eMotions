from __future__ import print_function, division
import argparse
import os
import sys
import subprocess

DEFAULT_CLASSES = ["Excitation", "Fear", "Neutral", "Relaxation", "Sadness", "Tension"]


def class_process(dir_path, dst_dir_path, class_name):

    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_dir_path):
        os.mkdir(dst_class_path)


    for file_name in os.listdir(class_path):

        name, ext          = os.path.splitext(file_name)
        dst_directory_path = os.path.join(dst_class_path, name)
        video_file_path    = os.path.join(class_path, file_name)

        try:
            if os.path.exists(dst_directory_path):
                if not os.path.exists(os.path.join(dst_directory_path, 'image00001.jpg')):
                    subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                    print('remove {}'.format(dst_directory_path))
                    os.makedirs(dst_directory_path)
                else:
                    continue
            else:
                os.mkdir(dst_directory_path)

        except:
            print(dst_directory_path)
            continue

        cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/%06d.jpg\"'.format(video_file_path, dst_directory_path)
        print(cmd)
        subprocess.call(cmd, shell=True)
        print('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract jpg frames from class-organized video folders.")
    parser.add_argument("--input_dir", required=True, help="Root directory of raw videos organized by class.")
    parser.add_argument("--output_dir", required=True, help="Output root directory for extracted frames.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="Class names to process.")
    args = parser.parse_args()

    for class_name in args.classes:
        class_process(args.input_dir, args.output_dir, class_name)
