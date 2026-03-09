import os
import numpy as np
from scipy.io.wavfile import write
import torch
import random
from importlib import import_module
from omegaconf import OmegaConf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "./speech_tokenization/UVITS"))
import uvits_utils
from emova_speech_tokenizer.speech_tokenization.UVITS.models import SynthesizerTrn
from text import text_to_sequence
from my_synthesis.my_synthesis_for_speech_unit_sequence_recombination import get_U2S_config_checkpoint_file
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "./speech_tokenization/SPIRAL_L2_BN_FSQ_CTC"))
from my_extract_unit_for_speech.extract_unit_construct_wav_unit_text import \
    get_S2U_ckpt_config_path, sample_extract_unit, batch_extract_unit
from nemo.collections.asr.models.spec2vec.vq_ctc_finetune import VQCTCFinetuneModel
from nemo.utils import logging

#################
# S2U
#################
def load_config(config=None):
    if config is not None:
        print("Config: ", config)
        cfg_module = import_module(config.replace('/', '.'))
    cfg = OmegaConf.structured(cfg_module.cfg)
    OmegaConf.set_struct(cfg, True)
    return cfg

def load_S2U_model(ckpt_path, config_path, model_name):
    assert model_name in ['SPIRAL-FSQ-CTC']
    cfg = load_config(config=config_path)
    cfg.model.pretrain_chkpt_path = None
    model = VQCTCFinetuneModel(cfg=cfg.model, trainer=None).eval()
    model = model.to(dtype=torch.float32)
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), ckpt_path), map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
    if(missing_keys):
        logging.warning('Missing Keys: {}'.format(missing_keys))
    if(unexpected_keys):
        logging.warning('Unexpected Keys: {}'.format(unexpected_keys))    
    return model

def s2u_extract_unit_demo(model, wav_path, model_name, reduced=True):
    assert model_name in ['SPIRAL-FSQ-CTC']

    wav_file_list = [wav_path]
    wav_file_list_len = 1
    extracted_wav_file_list, skipped_wav_file_list, unreduced_unit_sequence_list, reduced_unit_sequence_list = batch_extract_unit(wav_file_list, model, max_chunk=960000)
    target_unit_sequence_list = reduced_unit_sequence_list if reduced else unreduced_unit_sequence_list
    if len(extracted_wav_file_list) != 0:
        target_unit_sequence = target_unit_sequence_list[0]
    else:
        wav_file = skipped_wav_file_list[0]
        unreduced_unit_sequence, reduced_unit_sequence = sample_extract_unit(wav_file, model)
        target_unit_sequence = reduced_unit_sequence if reduced else unreduced_unit_sequence

    return "".join(["<|speech_{}|>".format(each) for each in target_unit_sequence.split(" ")])

#################
# U2S
#################
def load_condition_centroid(condition2style_centroid_file):
    with open(os.path.join(os.path.dirname(__file__), condition2style_centroid_file), 'r') as f:
        line_list = [line.replace('\n', '') for line in f]
    assert line_list[0] == 'condition|style_centroid_file'
    condition2style_centroid_file_dict, condition2style_centroid_embedding_dict = {}, {}
    for line in line_list[1:]:
        condition, style_centroid_file = line.split('|')
        condition2style_centroid_file_dict[condition] = style_centroid_file
        style_centroid_embedding = np.load(os.path.join(os.path.dirname(__file__), style_centroid_file))
        style_centroid_embedding = torch.FloatTensor(style_centroid_embedding).unsqueeze(1).unsqueeze(0)
        condition2style_centroid_embedding_dict[condition] = style_centroid_embedding
    return condition2style_centroid_file_dict, condition2style_centroid_embedding_dict
    
def load_U2S_config(model_config_file):
    hps = uvits_utils.get_hparams_from_file(os.path.join(os.path.dirname(__file__), model_config_file))
    from text.symbols import symbols_with_4096 as symbols
    hps.num_symbols = len(symbols)
    return hps

def load_U2S_model(model_config_file, model_checkpoint_file, unit_type, ):
    # load model
    hps = uvits_utils.get_hparams_from_file(os.path.join(os.path.dirname(__file__), model_config_file))
    from text.symbols import symbols_with_4096 as symbols
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    net_g.eval()
    uvits_utils.load_checkpoint(os.path.join(os.path.dirname(__file__), model_checkpoint_file), net_g, None)
    return net_g, hps

def synthesis(unit_sequence, style_embedding, hps, net_g, output_wav_file='output.wav'):
    # synthesize speech
    device = next(net_g.parameters()).device # we assume speech tokenizer is stored in a single device
    logging.info("Generating audios on {}".format(device))
    with torch.no_grad():
        unit_sequence = text_to_sequence(unit_sequence, hps.data.text_cleaners)
        unit_sequence = torch.LongTensor(unit_sequence)
        unit_sequence = unit_sequence.unsqueeze(0).to(device)
        unit_lengths = torch.LongTensor([unit_sequence.size(1)]).to(device)
        if style_embedding is not None:
            style_embedding = style_embedding.to(device)
        audio = net_g.synthesis_from_content_unit_style_embedding(
            unit_sequence, unit_lengths, style_embedding,
            noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0, 0].data.cpu().float().numpy()
    write(output_wav_file, hps.data.sampling_rate, audio)
    print(f'synthesized sample is saved as {output_wav_file}')
    return audio

if __name__ == "__main__":
    #################
    # NPU
    #################
    try:
        import torch_npu
        from torch_npu.npu import amp
        from torch_npu.contrib import transfer_to_npu
        print('Successful import torch_npu')
    except Exception as e:
        print(e)
    
    ############
    # S2U
    ############
    reduced = True
    reduced_mark = 'reduced' if reduced else 'unreduced'
    unit_type = '40ms_multilingual_8888'
    S2U_model_name = 'SPIRAL-FSQ-CTC'

    S2U_ckpt_path, S2U_config_path = get_S2U_ckpt_config_path(unit_type)
    S2U_model = load_S2U_model(S2U_ckpt_path, S2U_config_path, S2U_model_name)
    S2U_model = S2U_model.cuda()

    wav_file = "./examples/s2u/example.wav"
    speech_unit = s2u_extract_unit_demo(S2U_model, wav_file, model_name=S2U_model_name, reduced=reduced)
    print(speech_unit)
    
    ############
    # U2S
    ############
    condition2style_centroid_file = "./speech_tokenization/condition_style_centroid/condition2style_centroid.txt"
    condition2style_centroid_file_dict, condition2style_centroid_embedding_dict = load_condition_centroid(condition2style_centroid_file)

    unit_type = '40ms_multilingual_8888_xujing_cosyvoice_FT'
    U2S_config_file, U2S_checkpoint_file = get_U2S_config_checkpoint_file(unit_type)
    net_g, hps = load_U2S_model(U2S_config_file, U2S_checkpoint_file, unit_type)
    net_g = net_g.cuda()
    
    content_unit = speech_unit.replace('<|speech_', '').replace('|>', ' ').strip()
    emotion = random.choice(['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised'])
    speed = random.choice(['normal', 'fast', 'slow'])
    pitch = random.choice(['normal', 'high', 'low'])
    gender = random.choice(['female', 'male'])
    condition = f'gender-{gender}_emotion-{emotion}_speed-{speed}_pitch-{pitch}'

    style_centroid_file = condition2style_centroid_file_dict[condition]
    style_centroid_embedding = condition2style_centroid_embedding_dict[condition]

    output_wav_file = f'./examples/u2s/{condition}_output.wav'
    synthesis(content_unit, style_centroid_embedding, hps, net_g, output_wav_file)
