from models.AVNet import AVNet


def generate_model(opt):
    model = AVNet(
        snippet_duration          = opt.snippet_duration,
        sample_size               = opt.sample_size,
        n_classes                 = opt.n_classes,
        seq_len                   = opt.seq_len,
        audio_embed_size          = opt.audio_embed_size,
        audio_n_segments          = opt.audio_n_segments,
        pretrained_video_path     = opt.video_pretrained,
        audio_pretrained          = getattr(opt, 'audio_pretrained', True),
    )
    model = model.cuda()
    return model, model.parameters()
