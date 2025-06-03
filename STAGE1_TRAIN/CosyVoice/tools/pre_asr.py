import argparse
import logging
import os

import torch
from tqdm import tqdm
from transformers import pipeline
import torchaudio
from accelerate import PartialState
from accelerate.utils import gather_object

import transformers
transformers.models.whisper.tokenization_whisper.logger.setLevel(logging.ERROR)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def _gather_results(out_path, is_main_process):
    utt2asr = []
    for i in range(torch.cuda.device_count()):
        try:
            path = out_path + f'.list.cuda:{i}'
            utt2asr += torch.load(path)
        except:
            continue
    utt2asr = {k: v for k, v in utt2asr}
    try:
        reminded = torch.load(out_path)
        utt2asr.update(reminded)
    except:
        pass

    if is_main_process:
        print('size of utt2asr:', len(utt2asr))
        torch.save(utt2asr, out_path)
    return utt2asr


def main(args):
    distributed_state = PartialState()

    out_path = '{}/utt2asr.pt'.format(args.dir)
    utt2asr = _gather_results(out_path, distributed_state.is_main_process)

    utt2wav = {}
    with open('{}/wav.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]

    utt_list = [utt for utt in utt2wav.keys() if utt not in utt2asr]

    if len(utt_list) == 0:
        if distributed_state.is_main_process:
            print(f'The job is finish on {out_path}.')
        return

    pipe = pipeline(
        'automatic-speech-recognition',
        model=args.whisper_path,
        torch_dtype=torch.float16,
        device=distributed_state.device,
        chunk_length_s=30,
    )
    save_iter = 10
    with distributed_state.split_between_processes(utt_list) as sub_utt_list:
        def _wave_generator(utt_list, utt2wav):
            for utt in tqdm(utt_list):
                yield utt2wav[utt]

        # sorted by audio length
        audio_lengths = {}
        for utt in tqdm(sub_utt_list, desc='get audio lengths'):
            metadata = torchaudio.info(utt2wav[utt])
            audio_length = metadata.num_frames / metadata.sample_rate
            audio_lengths[utt] = audio_length
        sub_utt_list = sorted(sub_utt_list, key=lambda utt: audio_lengths[utt])

        generator = _wave_generator(sub_utt_list, utt2wav)

        utt2asr_queue = []
        for result, utt in zip(pipe(
                    generator,
                    return_timestamps='word',
                    generate_kwargs={
                        'language': 'english',
                        'forced_decoder_ids': None,
                        'task': 'transcribe'
                    },
                    batch_size=args.bs,
                ), sub_utt_list):

            try:
                sample = {}
                sample['asr_text'] = result['text']
                sample['asr_chunks'] = [
                    {'word': x['text'],
                    'range': (
                        x['timestamp'][0] / audio_lengths[utt],
                        x['timestamp'][1] / audio_lengths[utt]
                    )}
                    for x in result['chunks']
                ]
            except:
                print('Error: ASR Failed')
                sample = {
                    'asr_text': '<unk>',
                    'asr_chunks': [{'word': '<unk>', 'range': (0., 1.)}]
                }
            utt2asr_queue.append([utt, sample])
            if len(utt2asr_queue) % save_iter == save_iter - 1:
                torch.save(utt2asr_queue, out_path + f'.list.{distributed_state.device}')
        torch.save(utt2asr_queue, out_path + f'.list.{distributed_state.device}')

    distributed_state.wait_for_everyone()

    if distributed_state.is_main_process:
        _gather_results(out_path, distributed_state.is_main_process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        type=str)
    parser.add_argument('--bs',
                        type=int, default=1)
    parser.add_argument('--whisper_path',
                        type=str, default='/proj/mtklmadm/models/whisper-large-v3')
    args = parser.parse_args()
    main(args)
