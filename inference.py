import argparse
import json
import os
from functools import partial
from typing import Union
from tqdm import tqdm

import librosa
import numpy as np
import soundfile as sf
import torch

from loudness_norm import loudness_norm
from loguru import logger
from mmengine import Config

from feature_extractors import FEATURE_EXTRACTORS, PITCH_EXTRACTORS
from audio_processing import get_mel_from_audio, slice_audio
from utils import load_checkpoint

from tensor import repeat_expand

@torch.no_grad()
def inference(
    config,
    checkpoint,
    input_path,
    output_path,
    speaker_id=0,
    pitch_adjust=0,
    silence_threshold=60,
    max_slice_duration=30.0,
    vocals_loudness_gain=0.0,
    sampler_interval=None,
    sampler_progress=False,
    device="cuda",
):
    """Inference

    Args:
        config: config
        checkpoint: checkpoint path
        input_path: input path
        output_path: output path
        speaker_id: speaker id
        pitch_adjust: pitch adjust
        silence_threshold: silence threshold of librosa.effects.split
        max_slice_duration: maximum duration of each slice
        extract_vocals: extract vocals
        merge_non_vocals: merge non-vocals, only works when extract_vocals is True
        vocals_loudness_gain: loudness gain of vocals (dB)
        sampler_interval: sampler interval, lower value means higher quality
        sampler_progress: show sampler progress
        device: device
        gradio_progress: gradio progress callback
    """

    if sampler_interval is not None:
        config.model.diffusion.sampler_interval = sampler_interval

    if os.path.isdir(checkpoint):
        # Find the latest checkpoint
        checkpoints = sorted(os.listdir(checkpoint))
        logger.info(f"Found {len(checkpoints)} checkpoints, using {checkpoints[-1]}")
        checkpoint = os.path.join(checkpoint, checkpoints[-1])

    audio, sr = librosa.load(input_path, sr=config.sampling_rate, mono=True)

    # Normalize loudness
    audio = loudness_norm(audio, sr)

    # Slice into segments
    segments = list(
        slice_audio(
            audio, sr, max_duration=max_slice_duration, top_db=silence_threshold
        )
    )
    logger.info(f"Sliced into {len(segments)} segments")

    # Load models
    text_features_extractor = FEATURE_EXTRACTORS.build(
        config.preprocessing.text_features_extractor
    ).to(device)
    text_features_extractor.eval()

    model = load_checkpoint(config, checkpoint, device=device)

    pitch_extractor = PITCH_EXTRACTORS.build(config.preprocessing.pitch_extractor)
    assert pitch_extractor is not None, "Pitch extractor not found"

    generated_audio = np.zeros_like(audio)
    audio_torch = torch.from_numpy(audio).to(device)[None]

    for idx, (start, end) in enumerate(tqdm(segments, desc='Generating audio...')):
        segment = audio_torch[:, start:end]
        logger.info(
            f"Processing segment {idx + 1}/{len(segments)}, duration: {segment.shape[-1] / sr:.2f}s"
        )

        # Extract mel
        mel = get_mel_from_audio(segment, sr)

        # Extract pitch (f0)
        pitch = pitch_extractor(segment, sr, pad_to=mel.shape[-1]).float()
        pitch *= 2 ** (pitch_adjust / 12)

        # Extract text features
        text_features = text_features_extractor(segment, sr)[0]
        text_features = repeat_expand(text_features, mel.shape[-1]).T

        # Predict
        src_lens = torch.tensor([mel.shape[-1]]).to(device)

        features = model.model.forward_features(
            speakers=torch.tensor([speaker_id]).long().to(device),
            contents=text_features[None].to(device),
            src_lens=src_lens,
            max_src_len=max(src_lens),
            mel_lens=src_lens,
            max_mel_len=max(src_lens),
            pitches=pitch[None].to(device),
        )

        result = model.model.diffusion(features["features"], progress=sampler_progress)
        wav = model.vocoder.spec2wav(result[0].T, f0=pitch).cpu().numpy()
        max_wav_len = generated_audio.shape[-1] - start
        generated_audio[start : start + wav.shape[-1]] = wav[:max_wav_len]


    # Loudness normalization
    generated_audio = loudness_norm(generated_audio, sr)

    # Loudness gain
    loudness_float = 10 ** (vocals_loudness_gain / 20)
    generated_audio = generated_audio * loudness_float

    logger.info("Done")

    if output_path is not None:
        sf.write(output_path, generated_audio, sr)

    return generated_audio, sr


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Path to the input audio file",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to the output audio file",
    )

    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="Speaker id",
    )

    parser.add_argument(
        "--speaker_mapping",
        type=str,
        default=None,
        help="Speaker mapping file (gradio mode only)",
    )

    parser.add_argument(
        "--pitch_adjust",
        type=int,
        default=0,
        help="Pitch adjustment in semitones",
    )

    parser.add_argument(
        "--vocals_loudness_gain",
        type=float,
        default=0,
        help="Loudness gain for vocals",
    )

    parser.add_argument(
        "--sampler_interval",
        type=int,
        default=None,
        required=False,
        help="Sampler interval, if not specified, will be taken from config",
    )

    parser.add_argument(
        "--sampler_progress",
        action="store_true",
        help="Show sampler progress",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        required=False,
        help="Device to use",
    )

    return parser.parse_args()

def run_inference(
    config_path: str,
    model_path: str,
    input_path: str,
    speaker: Union[int, str],
    pitch_adjust: int,
    sampler_interval: int,
    device: str,
    speaker_mapping: dict = None,
):
    if speaker_mapping is not None and isinstance(speaker, str):
        speaker = speaker_mapping[speaker]

    audio, sr = inference(
        Config.fromfile(config_path),
        model_path,
        input_path=input_path,
        output_path=None,
        speaker_id=speaker,
        pitch_adjust=pitch_adjust,
        sampler_interval=round(sampler_interval),
        merge_non_vocals=False,
        device=device,
    )
    return (sr, audio)

if __name__ == "__main__":
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    inference(
        Config.fromfile(args.config),
        args.checkpoint,
        args.input,
        args.output,
        speaker_id=args.speaker_id,
        pitch_adjust=args.pitch_adjust,
        vocals_loudness_gain=args.vocals_loudness_gain,
        sampler_interval=args.sampler_interval,
        sampler_progress=args.sampler_progress,
        device=device
    )
