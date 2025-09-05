import argparse
import json
import os

import numpy as np
import torch
import torchaudio

from models.ae import OobleckEncoder, OobleckDecoder, AudioAutoencoder
from models.bottleneck import RoundBottleneck
from models.lm_continuous import LaplaceLanguageModel


# Fixed evaluation settings
SR = 44100
IN_CHANNELS = 2
OUT_CHANNELS = 2
LATENT_DIM = 2048
ENC_CHANNELS = 64
STRIDES = [2, 4, 4, 4]  # downsampling_ratio = 128
SAMPLE_SIZE = 32768      # training window in samples
LM_DEPTH = 8
LM_DIM_HEADS = 64
WEIGHTS_DIR = "weights"
OUT_DIR = "outputs"


def load_audio(path: str, target_sr: int, target_len: int, device: torch.device) -> torch.Tensor:
    wav, sr = torchaudio.load(path)  # [C, N]
    wav = wav.to(device)
    # force stereo (2 channels)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2, :]
    # resample to target_sr (e.g., 44100)
    if sr != target_sr:
        resamp = torchaudio.transforms.Resample(sr, target_sr).to(device)
        wav = resamp(wav)
    wav = wav.clamp(-1, 1).unsqueeze(0)  # [1, 2, N]
    N = wav.shape[-1]
    if N < target_len:
        wav = torch.nn.functional.pad(wav, (0, target_len - N))
    else:
        wav = wav[..., :target_len]
    return wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build models: AE + Round bottleneck + continuous LM
    enc = OobleckEncoder(in_channels=IN_CHANNELS, channels=ENC_CHANNELS, latent_dim=LATENT_DIM, c_mults=[1,2,4,8], strides=STRIDES, use_snake=True).to(device)
    dec = OobleckDecoder(out_channels=OUT_CHANNELS, channels=ENC_CHANNELS, latent_dim=LATENT_DIM, c_mults=[1,2,4,8], strides=STRIDES, use_snake=True, final_tanh=False).to(device)
    dsr = 1
    for s in STRIDES:
        dsr *= s
    bottleneck = RoundBottleneck(latent_dim=LATENT_DIM)
    ae = AudioAutoencoder(
        encoder=enc,
        decoder=dec,
        latent_dim=LATENT_DIM,
        downsampling_ratio=dsr,
        sample_rate=SR,
        io_channels=OUT_CHANNELS,
        bottleneck=bottleneck,
    ).to(device)
    lm_cfg = {"backbone": {"depth": LM_DEPTH, "dim_heads": LM_DIM_HEADS}}
    lm = LaplaceLanguageModel(LATENT_DIM, lm_cfg).to(device)

    # Load weights if provided
    def maybe_load(module, path):
        if os.path.exists(path):
            module.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded {path}")

    maybe_load(enc, os.path.join(WEIGHTS_DIR, "ae_encoder.pth"))
    maybe_load(dec, os.path.join(WEIGHTS_DIR, "ae_decoder.pth"))
    maybe_load(lm, os.path.join(WEIGHTS_DIR, "lm.pth"))

    # IO
    target_len = SAMPLE_SIZE
    audio = load_audio(args.audio, SR, target_len, device)

    # Encode (include bottleneck -> integer tokens)
    with torch.no_grad():
        latents_q = ae.encode(audio)
    np.save(os.path.join(OUT_DIR, "tokens.npy"), latents_q.squeeze(0).to(torch.int64).cpu().numpy())

    # Continuous LM: µ,b over quantized latents (as used in your training wrapper)
    with torch.no_grad():
        mu, b = lm(latents_q)
    np.save(os.path.join(OUT_DIR, "mu.npy"), mu.squeeze(0).cpu().numpy())
    np.save(os.path.join(OUT_DIR, "b.npy"), b.squeeze(0).cpu().numpy())

    # Discrete mass and bitrate
    # Laplace CDF(x) = 0.5 - 0.5 * sign(x-mu) * expm1(-|x-mu|/b)
    x_plus = latents_q + 0.5
    x_minus = latents_q - 0.5
    cdf_plus = 0.5 - 0.5 * (x_plus - mu).sign() * torch.expm1(- (x_plus - mu).abs() / b)
    cdf_minus = 0.5 - 0.5 * (x_minus - mu).sign() * torch.expm1(- (x_minus - mu).abs() / b)
    p = torch.clamp_min(cdf_plus - cdf_minus, 1e-6)
    np.save(os.path.join(OUT_DIR, "probs.npy"), p.squeeze(0).cpu().numpy())
    rate_bits = torch.clamp(-torch.log2(p), max=12)
    # per-scalar average -> per-token (sum over channels)
    bits_per_scalar = rate_bits.mean().item()
    C = int(latents_q.shape[1])
    bits_per_token = bits_per_scalar * C
    tokens_per_sec = SR / dsr
    bits_per_sec = bits_per_token * tokens_per_sec
    with open(os.path.join(OUT_DIR, "rate.json"), "w") as f:
        json.dump({
            "bits_per_token": bits_per_token,
            "tokens_per_sec": tokens_per_sec,
            "bits_per_sec": bits_per_sec,
        }, f, indent=2)

    # Decode from integer latents
    with torch.no_grad():
        recon = ae.decode(latents_q)
    torchaudio.save(os.path.join(OUT_DIR, "reconstruction.wav"), recon.squeeze(0).cpu().clamp(-1, 1), SR)

    print("Done.")


if __name__ == "__main__":
    main()


