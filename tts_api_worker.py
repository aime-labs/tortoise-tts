import argparse
import os
import io
from collections import deque

import torch
import torchaudio

from tortoise.api import TextToSpeech
from tortoise.api_fast import TextToSpeech as TextToSpeechFast
from tortoise.utils.audio import load_voices
from tortoise.utils.text import split_and_recombine_text
from aime_api_worker_interface import APIWorkerInterface

MODELS_DIR = './models'

WORKER_JOB_TYPE = "tts_tortoise"
DEFAULT_WORKER_AUTH_KEY = "5317e305b50505ca2b3284b4ae5f65a5"
VERSION = 0

PRESETS = {
    'ultra_fast': {'num_autoregressive_samples': 1, 'diffusion_iterations': 10},
    'fast': {'num_autoregressive_samples': 32, 'diffusion_iterations': 50},
    'standard': {'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
    'high_quality': {'num_autoregressive_samples': 256, 'diffusion_iterations': 400},
}

def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_deepspeed', action='store_true', help='Use deepspeed for speed bump.')
    parser.add_argument(
        '--kv_cache', type=bool, help='If you disable this please wait for a long time to get the output', default=True
    )
    parser.add_argument(
        '--half', type=bool, help="float16(half) precision inference if True it's faster and take less vram and ram", default=True
    )
    parser.add_argument(
        '--model_dir', type=str, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to .models, so this'
        'should only be specified if you have custom checkpoints.', default=MODELS_DIR
    )
    parser.add_argument(
        '--seed', type=int, help='Random seed which can be used to reproduce results.', default=None)
    parser.add_argument(
        '--produce_debug_state', action='store_true', help='Whether or not to produce debug_state.pth, which can aid in reproducing problems. Defaults to true.', default=False
    )
    parser.add_argument(
        '--cvvp_amount', type=float, help='How much the CVVP model should influence the output.'
        'Increasing this can in some cases reduce the likelihood of multiple speakers. Defaults to 0 (disabled)', default=.0
    )
    parser.add_argument(
        "--api_server", type=str, default="http://0.0.0.0:7777", help="Address of the AIME API server"
    )
    parser.add_argument(
        "--api_auth_key", type=str , default=DEFAULT_WORKER_AUTH_KEY, required=False, help="API server worker auth key"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, required=False, help="ID of the GPU to be used"
    )
    parser.add_argument(
        "--stream", action='store_true', help="Use streaming"
    )
    return parser.parse_args()


def main():
    args = get_flags()
    if torch.backends.mps.is_available():
        args.use_deepspeed = False
    if args.stream:
        tts = TextToSpeechFast(models_dir=args.model_dir, use_deepspeed=args.use_deepspeed, kv_cache=args.kv_cache, half=args.half)
    else:
        tts = TextToSpeech(models_dir=args.model_dir, use_deepspeed=args.use_deepspeed, kv_cache=args.kv_cache, half=args.half, device_only=True)
    
    candidates = 1

    api_worker = APIWorkerInterface(args.api_server, WORKER_JOB_TYPE, args.api_auth_key, args.gpu_id, world_size=1, rank=0, gpu_name=torch.cuda.get_device_name(), worker_version=VERSION)
    
    while True:
        try:
            job_data = api_worker.job_request()
            print(f'Processing job {job_data.get("job_id")}...', end='', flush=True)

            selected_voice = job_data.get('voice')           

            preset = job_data.get('preset')
            print("Loading voice...")
            voice_samples, conditioning_latents = load_voices([selected_voice])
            output = {'model_name': 'tortoise_tts'}
            if args.stream:
                text_chunk_queue = deque(
                    split_and_recombine_text(job_data.get('text'))
                )
                counter = 0
                while text_chunk_queue:
                    try:
                        if api_worker.jobs_canceled and all(api_worker.jobs_canceled):
                            print('Canceled')
                            break
                        if any(api_worker.progress_input_params):
                            text_chunk_queue.append(api_worker.progress_input_params.pop(0)[0])
                    except AttributeError:
                        print('Legacy mode! Update api worker interface to receive progress input parameters')
                    stream = tts.tts_stream(
                        text_chunk_queue.popleft(),
                        voice_samples=voice_samples,
                        conditioning_latents=conditioning_latents,
                        k=candidates,
                        verbose=True,
                        use_deterministic_seed=args.seed,
                        return_deterministic_state=False, 
                        overlap_wav_len=1024, 
                        stream_chunk_size=job_data.get('stream_chunk_size'),
                        temperature=job_data.get('temperature'), 
                        length_penalty=job_data.get('length_penalty'), 
                        repetition_penalty=job_data.get('repetition_penalty'), 
                        top_p=job_data.get('top_p'),
                        max_mel_tokens=job_data.get('max_mel_tokens'),
                        cvvp_amount=job_data.get('cvvp_amount') or args.cvvp_amount,
                        cond_free=True, 
                        cond_free_k=2, 
                        diffusion_temperature=1.0,
                        **PRESETS.get(preset)
                    )
                    for audio_chunk, text_chunk in stream:
                        try:
                            if api_worker.jobs_canceled and all(api_worker.jobs_canceled):
                                print('Canceled')
                                break
                            if api_worker.progress_input_params:
                                text_chunk_queue.append(api_worker.progress_input_params.pop(0)[0])
                        except AttributeError:
                            pass
                        counter += 1
                        with io.BytesIO() as buffer:
                            torchaudio.save(
                                buffer,
                                audio_chunk.unsqueeze(0).cpu(),
                                format='wav',
                                sample_rate=24000,
                            )
                            output['audio_output'] = buffer
                            output['text_output'] = text_chunk
                            while True:
                                if api_worker.progress_data_received:
                                    break
                            try:
                                api_worker.stream_progress(counter, output)
                            except AttributeError:
                                api_worker.send_progress(counter, output)
                while True:
                    if api_worker.progress_data_received:
                        break
                        
                api_worker.send_job_results({'model_name': 'tortoise_tts'})

            else:
                text = job_data.get('text')
                gen, dbg_state = tts.tts_with_preset(text, k=candidates, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                            preset=preset, use_deterministic_seed=args.seed, return_deterministic_state=True, cvvp_amount=args.cvvp_amount)
                
                with io.BytesIO() as buffer:
                    torchaudio.save(
                        buffer,
                        gen.squeeze(0).cpu(),
                        format='wav',
                        sample_rate=24000,
                    )
                    output['audio_output'] = buffer
                    api_worker.send_job_results(output)        

        except ValueError as exc:
            print('Error', exc)
            continue


def split_text(text, max_length=200):
    doc = nlp(text)
    chunks = []
    chunk = []
    length = 0

    for sent in doc.sents:
        sent_length = len(sent.text)
        if length + sent_length > max_length:
            chunks.append(' '.join(chunk))
            chunk = []
            length = 0
        chunk.append(sent.text)
        length += sent_length + 1

    if chunk:
        chunks.append(' '.join(chunk))

    return chunks


if __name__ == '__main__':
    main()