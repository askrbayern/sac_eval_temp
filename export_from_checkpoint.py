import argparse
import os
import torch

# this maps checkpoint to weights under weights/
# sample ussage:
# python export_from_checkpoint.py   --ckpt .../last.ckpt   --out_dir weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to big checkpoint (.ckpt or .safetensors)")
    parser.add_argument("--out_dir", type=str, default="weights")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load big state dict
    if args.ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(args.ckpt)
    else:
        obj = torch.load(args.ckpt, map_location=args.device)
        sd = obj.get("state_dict", obj)

    # TODO: map keys from your full model to minrepo modules
    # Below is a template. Adjust the prefixes to match your real checkpoint.
    enc_sd = {}
    dec_sd = {}
    lm_sd = {}

    for k, v in sd.items():
        # Example heuristics – replace with your naming scheme
        if k.startswith("model.encoder") or k.startswith("autoencoder.encoder"):
            enc_sd[k.split(".", 2)[-1]] = v
        elif k.startswith("model.decoder") or k.startswith("autoencoder.decoder"):
            dec_sd[k.split(".", 2)[-1]] = v
        elif k.startswith("lm.") or k.startswith("model.lm"):
            lm_sd[k.split(".", 1)[-1]] = v

    if enc_sd:
        torch.save(enc_sd, os.path.join(args.out_dir, "ae_encoder.pth"))
        print("Wrote ae_encoder.pth")
    if dec_sd:
        torch.save(dec_sd, os.path.join(args.out_dir, "ae_decoder.pth"))
        print("Wrote ae_decoder.pth")
    if lm_sd:
        torch.save(lm_sd, os.path.join(args.out_dir, "lm.pth"))
        print("Wrote lm.pth")

    print("Done exporting.")


if __name__ == "__main__":
    main()


