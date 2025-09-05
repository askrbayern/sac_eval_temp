# sac_eval_temp (eval-only AE+LM)
## Structure of Codebase
- models/ae.py: Encoder/Decoder (conv1d) and round-only FSQ quantizer (encode/decode_tokens)
- models/lm.py: Minimal token LM (per-quantizer embeddings + GRU)
- run.py: a script to encode and decode. It saves the latents and reload it from the outputs folder. It also passes the reloaded latent to language model to calculate and save b, mu, probability, and rate.
- export_from_checkpoint.py: this loads a checkpoint and export it to weights folder. You dont need this. I ran it using a epoch 50 checkpoint already.

## Things to do
- it should be runnable with any environment that has torch torchaudio numpy einops
- download weights and extract it under weights, thre should be ae_decoder.pth, ae_encoder.pth, lm.pth
- download a sample audio (jazz.mp3) in the main folder, or any other music, it doesnt matter, as the clip deterministically takes the first ~0.74s of any audio)

## sample usage
Code for turning checkpoint to weights
python export_from_checkpoint.py   --ckpt .../last.ckpt   --out_dir weights

Code for eval
python run.py --audio jazz.mp3

## Basic Parameters
- Fixed params: SR=44100, dsr=128, latent_dim=2048, stereo, window=32768 samples.
- Weights: place ae_encoder.pth, ae_decoder.pth, lm.pth under ./weights/
- Outputs: ./outputs/{reconstruction.wav, tokens.npy, mu.npy, b.npy, probs.npy, rate.json}
- Bitrate: bits_per_token = mean(-log2 p) * 2048; tokens_per_sec = SR/dsr; bits_per_sec = product.

## Output to outputs folder
- `latents.npy` – encoder output, by default quantized (rounded)
- `tokens.npy` – FSQ tokens [2048, 256], 2048 is the latent dim, 256 is the number of latents in one clipped sample
- `mu.npy`, `b.npy` – [2048, 256], obtained from LM
- `probs.npy` – [2048, 256], calculated from mu and b
- `rate.json` – bits per latent, number of latents of second, and bits_per_second (=bitrate)
- `reconstruction.wav` – decode from tokens, deleted from the repo
