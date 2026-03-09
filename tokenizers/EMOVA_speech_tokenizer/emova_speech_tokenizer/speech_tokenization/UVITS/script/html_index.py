from dataclasses import dataclass
import os
from argparse import ArgumentParser


def main(conf):
    texts = [v.strip().split('|') for v in open(conf.text_file)]
    if conf.n_speakers > 0:
        texts = {k: v for k, _, v in texts}
    else:
        texts = {k: v for k, v in texts}

    with open(conf.input_file) as f, open(conf.index_file, 'w') as g:
        g.write("Text| VITS-Ref| Ground Truth\n")
        for idx, li in enumerate(f):
            n, *_ = li.strip().split('|')
            text = texts[n]
            w1, w2 = os.path.join(conf.audio_dir, f'{idx}_pred.wav'), \
                os.path.join(conf.audio_dir, f'{idx}_gt.wav')
            g.write(f'{text}|{w1}|{w2}\n')


def vc_main(conf):
    texts = [v.strip().split('|') for v in open(conf.text_file)]
    texts = {k: v for k, _, v in texts}

    with open(conf.input_file) as f, open(conf.index_file, 'w') as g:
        g.write("Text| Source | Conversion(F) | Target(F) | Conversion(M) | Target(M) \n")
        for idx, li in enumerate(f):
            n, *_ = li.strip().split('|')
            text = texts[n]
            sr, vc1, tar1, vc2, tar2 = os.path.join(conf.audio_dir, f'{idx}_src.wav'), \
                os.path.join(conf.audio_dir, f'{idx}_vc0.wav'), \
                os.path.join(conf.audio_dir, f'{idx}_tar0.wav'), \
                os.path.join(conf.audio_dir, f'{idx}_vc1.wav'), \
                os.path.join(conf.audio_dir, f'{idx}_tar1.wav')
            g.write(f'{text}|{sr}|{vc1}|{tar1}|{vc2}|{tar2}\n')


if __name__ == '__main__':

    @dataclass
    class Conf:
        text_file: str = 'filelists/vctk_audio_sid_text_test_filelist.txt'
        input_file: str = 'filelists/vctk_audio_sid_text_test_filelist.txt.unit.reduced'
        index_file: str = 'result/index.txt'
        audio_dir: str = 'result/spiral-20ms/base'


    # conf = Conf()

    parser = ArgumentParser()
    parser.add_argument('--text_file', default=None)
    parser.add_argument('--input_file', default=None)
    parser.add_argument('--vc_file', default=None)
    parser.add_argument('--index_file', type=str, required=True)
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--n_speakers', type=int, default=109)
    args = parser.parse_args()

    if args.vc_file is None:
        main(args)
    else:
        vc_main(args)


    @dataclass
    class VCConf:
        text_file: str = 'filelists/vctk_audio_sid_text_test_filelist.txt'
        input_file: str = 'filelists/vctk_audio_sid_text_test_filelist.txt.unit.reduced'
        vc_file: str = 'filelists/vctk_vc_pairs.txt'
        index_file: str = 'result/index-vc.txt'
        audio_dir: str = 'result/spiral-20ms/base-vc'


    @dataclass
    class CmuJVCConf:
        text_file: str = 'filelists/cmu_arctic_audio_sid_text_test_filelist.txt'
        input_file: str = 'filelists/cmu_arctic_audio_sid_text_test_filelist.unit.reduced'
        vc_file: str = 'filelists/cmu_vctk_vc_pairs.txt'
        index_file: str = 'result/index-vc-a2m.txt'
        audio_dir: str = 'result/spiral-20ms/base-vc-a2m'


    @dataclass
    class CmuVCConf:
        text_file: str = 'filelists/cmu_arctic_audio_sid_text_test_filelist.txt'
        input_file: str = 'filelists/cmu_arctic_audio_sid_text_test_filelist.unit.reduced'
        vc_file: str = 'filelists/cmu_vc_pairs.txt'
        index_file: str = 'result/index-vc-a2a.txt'
        audio_dir: str = 'result/spiral-20ms/base-vc-a2a'


    @dataclass
    class CmuVCConfSC:
        text_file: str = 'filelists/cmu_arctic_audio_sid_text_test_filelist.txt'
        input_file: str = 'filelists/cmu_arctic_audio_sid_text_test_filelist.unit.reduced'
        vc_file: str = 'filelists/cmu_vc_pairs.txt'
        index_file: str = 'result/index-sc-vc-a2a.txt'
        audio_dir: str = 'result/spiral-20ms/sc-vc-a2a'

    # conf = VCConf()
    # conf = CmuJVCConf()
    # conf = CmuVCConf()
    # conf = CmuVCConfSC()
