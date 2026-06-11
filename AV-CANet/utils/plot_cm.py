import os
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, save_path, class_labels, title="Confusion Matrix", dpi=300):


    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truth")
    plt.yticks(range(class_labels.__len__()), class_labels)
    plt.xticks(range(class_labels.__len__()), class_labels, rotation=45)
    plt.tight_layout()
    plt.colorbar()

    for i in range(class_labels.__len__()):
        for j in range(class_labels.__len__()):
            color = (0, 0, 0)
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    if not save_path is None:
        plt.savefig(save_path, format='jpg', bbox_inches='tight', dpi=dpi)


def local2global_path(opt):
    if opt.root_path != '':
        opt.video_path      = os.path.join(opt.root_path, opt.video_path)
        opt.audio_path      = os.path.join(opt.root_path, opt.audio_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
    else:
        raise Exception
