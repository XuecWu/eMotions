\
\
\
\


import argparse


def parse_opts():
\
\

    parser    = argparse.ArgumentParser()
    arguments = {
        'coefficients': [
            dict(name    = '--lambda_POS',
                 default = '0.5',
                 type    = float,
                 help    = 'Penalty Coefficient that Controls the Penalty Extent of PSOITIVE in EP-CE Loss'),

            dict(name    = '--lambda_NEU',
                 default = '0.5',
                 type    = float,
                 help    = 'Penalty Coefficient that Controls the Penalty Extent of NEUTRAL in EP-CE Loss'),

            dict(name    = '--lambda_NEG',
                 default = '0.5',
                 type    = float,
                 help    = 'Penalty Coefficient that Controls the Penalty Extent of NEGATIVE in EP-CE Loss'),
        ],

        'paths': [
            dict(name    = '--video_pretrained',
                 default = 'pretrained/swin_tiny_patch4_window7_224_22k.pth',
                 type    = str,
                 help    = 'Global path of pretrained 3d resnet101 model (.pth)'),

            dict(name    = '--root_path',
                 default = ".",
                 type    = str,
                 help    = 'Global path of root directory'),

            dict(name    = "--video_path",
                 default = "eMotions_all_processed/eMotions_imgs",
                 type    = str,
                 help    = 'Local path of frames of videos', ),

            dict(name    = "--annotation_path",
                 default = 'annotations/eMotions6_01_all.json',
                 type    = str,
                 help    = 'Local path of annotation file'),

            dict(name    = "--result_path",
                 default = 'results',
                 type    = str,
                 help    = "Local path of result directory"),

            dict(name    = "--resume_path",
                 default = '',
                 type    = str,
                 help    = 'Checkpoint file path used in test/inference mode'),

            dict(name    = '--expr_name',
                 type    = str,
                 default = '0413-no-EP-CE'),

            dict(name    = '--audio_path',
                 type    = str,
                 default = 'eMotions_all_processed/eMotions_mp3',
                 help    = 'Local path of audios')
        ],

        'core': [
            dict(name    = '--batch_size',
                 default = 8,
                 type    = int,
                 help    = 'Batch Size'),

            dict(name    = '--snippet_duration',
                 default = 8,
                 type    = int),

            dict(name    = '--sample_size',
                 default = 112,
                 type    = int,
                 help    = 'Heights and width of inputs'),

            dict(name    = '--n_classes',
                 default = 6,
                 type    = int,
                 help    = 'Number of classes in the datasets'),

            dict(name    = '--seq_len',
                 default = 16,
                 type    = int),

            dict(name    = '--loss_func',
                 default = 'ce',
                 type    = str,
                 help    = 'ce | PCCE_Loss'),

            dict(name    = '--learning_rate',
                 default = 0.00015,
                 type    = float,
                 help    = 'Initial learning rate', ),

            dict(name    = '--fps',
                 default = 30,
                 type    = int,
                 help    = 'fps'),

            dict(name    = '--gpu_ids',
                 default = '0',
                 type    = str,
                 help    = 'GPU ids'
                 )
        ],

        'network': [
            {
                'name'   : '--audio_embed_size',
                'default': 512,
                'type'   : int,},
            {
                'name'   : '--audio_n_segments',
                'default': 16,
                'type'   : int,}
        ],

        'common': [
            dict(name    = '--dataset',
                 type    = str,
                 default = 've8',
                 ),
            dict(name    = '--use_cuda',
                 action  = 'store_true',
                 default = True,
                 help    = 'only cuda supported!'
                 ),
            dict(name    = '--debug',
                 default = False,
                 action  = 'store_true'
                 ),
            dict(name    = '--dl',
                 action  = 'store_true',
                 default = False,
                 help    = 'drop last'
                 ),
            dict(
                 name    = '--mode',
                 default = 'train',
                 type    = str,
                 choices = ['train', 'test'],
                 help    = 'which mode, train = train+val',
                ),
            dict(
                name     = '--num_workers',
                default  = 16,
                type     = int,
                help     = 'Number of workers in dataloader',
                ),
            dict(
                name     = '--n_epochs',
                default  = 30,
                type     = int,
                help     = 'Number of total epochs to run',
                )
        ],

        'OGM-GE': [
            dict(
                 name    = '--modulation',
                 default = 'Normal',
                 type    = str,
                 choices = ['Normal', 'OGM', 'OGM_GE'],
                 help    = 'Modulation manner',
            ),
            dict(
                 name    = '--fusion_method',
                 default = 'concat',
                 type    = str,
                 choices = ['sum', 'concat', 'gated', 'film'],
                 help    = 'Fusion method',
            ),
        ]
    }


    for group in arguments.values():

        for argument in group:
            name = argument['name']
            del argument['name']
            parser.add_argument(name, **argument)

    args = parser.parse_args()

    return args
