# sac_eval_temp (eval-only AE+LM)

Usage



- Fixed params: SR=44100, dsr=128, latent_dim=2048, stereo, window=32768 samples.
- Weights: place ae_encoder.pth, ae_decoder.pth, lm.pth under ./weights/
- Outputs: ./outputs/{reconstruction.wav, tokens.npy, mu.npy, b.npy, probs.npy, rate.json}
- Bitrate: bits_per_token = mean(-log2 p) * 2048; tokens_per_sec = SR/dsr; bits_per_sec = product.

