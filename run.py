import argparse
import json
import os

import numpy as np
import torch
import torchaudio

from models.ae import OobleckEncoder, OobleckDecoder, AudioAutoencoder
from models.bottleneck import RoundBottleneck
from models.lm_continuous import LaplaceLanguageModel


# Fixed evaluation settings for our 344Hz model
SR = 44100
IN_CHANNELS = 2
OUT_CHANNELS = 2
LATENT_DIM = 2048
ENC_CHANNELS = 64
STRIDES = [2, 4, 4, 4]  # downsampling_ratio = 128
SAMPLE_SIZE = 32768      # training window in samples, this is about 32768/44100 = 0.743 seconds
LM_DEPTH = 8
LM_DIM_HEADS = 64
WEIGHTS_DIR = "weights"
OUT_DIR = "outputs"
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_audio(path: str, target_sr: int, target_len: int, device: torch.device) -> torch.Tensor:
    wav, sr = torchaudio.load(path)  # [C, N]
    wav = wav.to(device)

    # force stereo (2 channels)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        print("Converting mono to stereo by doubling to two channels")
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2, :]
        print("Truncating to two channels")
    
    # resample to target_sr (e.g., 44100)
    if sr != target_sr:
        resamp = torchaudio.transforms.Resample(sr, target_sr).to(device)
        wav = resamp(wav)
        print(f"Resampled to {target_sr} Hz")

    wav = wav.clamp(-1, 1).unsqueeze(0)  # [1, 2, N]

    # padding
    N = wav.shape[-1]
    if N < target_len:
        wav = torch.nn.functional.pad(wav, (0, target_len - N))
    else:
        wav = wav[..., :target_len]
    
    print(f"Loaded audio with shape {wav.shape}")
    return wav


def load_weights(module, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required weights file not found: {path}")
    module.load_state_dict(torch.load(path, map_location=device))
    print(f"Loaded {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== Instantiate Models ====================

    # instantiate Encoder
    enc = OobleckEncoder(in_channels=IN_CHANNELS, channels=ENC_CHANNELS, latent_dim=LATENT_DIM, c_mults=[1,2,4,8], strides=STRIDES, use_snake=True).to(device)
    load_weights(enc, os.path.join(WEIGHTS_DIR, "ae_encoder.pth"))

    # instantiate Decoder
    dec = OobleckDecoder(out_channels=OUT_CHANNELS, channels=ENC_CHANNELS, latent_dim=LATENT_DIM, c_mults=[1,2,4,8], strides=STRIDES, use_snake=True, final_tanh=False).to(device)
    load_weights(dec, os.path.join(WEIGHTS_DIR, "ae_decoder.pth"))

    # instantiate Laplace Language Model
    lm_cfg = {"backbone": {"depth": LM_DEPTH, "dim_heads": LM_DIM_HEADS}}
    lm = LaplaceLanguageModel(LATENT_DIM, lm_cfg).to(device)
    load_weights(lm, os.path.join(WEIGHTS_DIR, "lm.pth"))



    # set downsampling ratio
    dsr = 1
    for s in STRIDES:
        dsr *= s

    # decide whether to use bottleneck, if it is None, the latent will be continuous values
    bottleneck = RoundBottleneck(latent_dim=LATENT_DIM)

    # wrap encoder and decoder into AudioAutoencoder
    ae = AudioAutoencoder(
        encoder=enc,
        decoder=dec,
        latent_dim=LATENT_DIM,
        downsampling_ratio=dsr,
        sample_rate=SR,
        io_channels=OUT_CHANNELS,
        bottleneck=bottleneck,
    ).to(device)



    # ============================ Evaluation ============================
    # 1. load audio
    target_len = SAMPLE_SIZE
    audio = load_audio(args.audio, SR, target_len, device)

    # 2. encode and save latents
    with torch.no_grad():
        saved_latents = ae.encode(audio) # by default quantized (rounded)
    np.save(os.path.join(OUT_DIR, "tokens.npy"), saved_latents.squeeze(0).to(torch.int64).cpu().numpy())
    print(f"Saved latents with shape {saved_latents.shape}, from {OUT_DIR}/tokens.npy")

    # 3. now load the saved tokens
    loaded_latents = torch.from_numpy(np.load(os.path.join(OUT_DIR, "tokens.npy"))).unsqueeze(0).to(device).float()
    print(f"Loaded latents with shape {loaded_latents.shape}, from {OUT_DIR}/tokens.npy")

    # 4. call LM and get µ,b over quantized latents, and save
    with torch.no_grad():
        mu, b = lm(loaded_latents)
    np.save(os.path.join(OUT_DIR, "mu.npy"), mu.squeeze(0).cpu().numpy())
    np.save(os.path.join(OUT_DIR, "b.npy"), b.squeeze(0).cpu().numpy())
    print(f"Saved µ,b with shape {mu.shape}, from {OUT_DIR}/mu.npy and {OUT_DIR}/b.npy")

    # 4.1 with these µ,b, we can calculate the probability of each token
    x_plus = loaded_latents + 0.5
    x_minus = loaded_latents - 0.5
    cdf_plus = 0.5 - 0.5 * (x_plus - mu).sign() * torch.expm1(- (x_plus - mu).abs() / b)
    cdf_minus = 0.5 - 0.5 * (x_minus - mu).sign() * torch.expm1(- (x_minus - mu).abs() / b)
    p = torch.clamp_min(cdf_plus - cdf_minus, 1e-6)
    np.save(os.path.join(OUT_DIR, "probs.npy"), p.squeeze(0).cpu().numpy())
    print(f"Saved probabilities with shape {p.shape}, from {OUT_DIR}/probs.npy")


    # 4.2 then we can calculate the bitrate
    rate_bits = torch.clamp(-torch.log2(p), max=12)
    bits_per_scalar = rate_bits.mean().item()
    C = int(loaded_latents.shape[1])
    assert C == LATENT_DIM, f"Expected latent_dim={LATENT_DIM}, got {C}"

    bits_per_latent = bits_per_scalar * C # this is per latent
    num_latents_per_sec = SR / dsr # which is 344
    bits_per_sec = bits_per_latent * num_latents_per_sec
    with open(os.path.join(OUT_DIR, "rate.json"), "w") as f:
        json.dump({
            "bits_per_latent": bits_per_latent,
            "num_latents_per_sec": num_latents_per_sec,
            "bits_per_sec": bits_per_sec,
        }, f, indent=2)

    print(f"Calculated bitrate {bits_per_sec}, from {OUT_DIR}/rate.json")

    # 5 Decode
    with torch.no_grad():
        recon = ae.decode(loaded_latents)
    torchaudio.save(os.path.join(OUT_DIR, "reconstruction.wav"), recon.squeeze(0).cpu().clamp(-1, 1), SR)
    print(f"Saved reconstruction with shape {recon.shape}, from {OUT_DIR}/reconstruction.wav")
    print("Whole process done.")



if __name__ == "__main__":
    main()


