import argparse
import os
import io

import torch
import torchaudio

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voices
from aime_api_worker_interface import APIWorkerInterface

MODELS_DIR = './models'

WORKER_JOB_TYPE = "tts_tortoise"
DEFAULT_WORKER_AUTH_KEY = "5317e305b50505ca2b3284b4ae5f65a5"
VERSION = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_deepspeed', type=str, help='Use deepspeed for speed bump.', default=False)
    parser.add_argument('--kv_cache', type=bool, help='If you disable this please wait for a long a time to get the output', default=True)
    parser.add_argument('--half', type=bool, help="float16(half) precision inference if True it's faster and take less vram and ram", default=True)
    parser.add_argument('--model_dir', type=str, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to .models, so this'
                                                      'should only be specified if you have custom checkpoints.', default=MODELS_DIR)
    parser.add_argument('--seed', type=int, help='Random seed which can be used to reproduce results.', default=None)
    parser.add_argument('--produce_debug_state', type=bool, help='Whether or not to produce debug_state.pth, which can aid in reproducing problems. Defaults to true.', default=False)
    parser.add_argument('--cvvp_amount', type=float, help='How much the CVVP model should influence the output.'
                                                          'Increasing this can in some cases reduce the likelihood of multiple speakers. Defaults to 0 (disabled)', default=.0)

    parser.add_argument("--api_server", type=str, default="http://0.0.0.0:7777", help="Address of the AIME API server")
    parser.add_argument("--api_auth_key", type=str , default=DEFAULT_WORKER_AUTH_KEY, required=False, help="API server worker auth key")
    parser.add_argument("--gpu_id", type=int, default=0, required=False, help="ID of the GPU to be used")

    args = parser.parse_args()

    if torch.backends.mps.is_available():
        args.use_deepspeed = False

    tts = TextToSpeech(models_dir=args.model_dir, use_deepspeed=args.use_deepspeed, kv_cache=args.kv_cache, half=args.half, device_only=True)

    selected_voice = "train_daws"
    candidates = 1

    api_worker = APIWorkerInterface(args.api_server, WORKER_JOB_TYPE, args.api_auth_key, args.gpu_id, world_size=1, rank=0, gpu_name=torch.cuda.get_device_name(), worker_version=VERSION)

    while True:
        try:
            job_data = api_worker.job_request()
            print(f'Processing job {job_data.get("job_id")}...', end='', flush=True)

            selected_voice = job_data.get('voice')
            text = job_data.get('text')
            preset = job_data.get('preset')

            print("Loading voice...")
            voice_samples, conditioning_latents = load_voices([selected_voice])

            gen, dbg_state = tts.tts_with_preset(text, k=candidates, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                        preset=preset, use_deterministic_seed=args.seed, return_deterministic_state=True, cvvp_amount=args.cvvp_amount)

            output = {'model_name': 'tortoise_tts'}
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
            print('Error')
#            callback.process_output(None , None, True, f'{exc}\nChange parameters and try again')
            continue
