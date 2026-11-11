# PAN 2 single-file models for ComfyUI

Scripts that turn the [IFM/PAN2](https://huggingface.co/IFM/PAN2) release (diffusers folders, about 81 GiB) into single
`.safetensors` files in ComfyUI's quantized formats.

## What you get

| Target name | File | Size | Notes |
|---|---|---|---|
| `transformer_int8_convrot` | `diffusion_models/PAN2_int8_convrot.safetensors` | 39.8 GiB | int8 ConvRot; the model's fp32 `norm1.linear` layers stay fp32 |
| `transformer_int8_convrot_full` | `diffusion_models/PAN2_int8_convrot_full.safetensors` | 27.4 GiB | int8 ConvRot, those layers quantized too |
| `transformer_nvfp4` | `diffusion_models/PAN2_nvfp4.safetensors` | 15.5 GiB | NVFP4 (4-bit); runs natively on RTX 50 series, emulated elsewhere |
| `te_int8_convrot` | `text_encoders/PAN2_qwen3_5_int8_convrot.safetensors` | 8.4 GiB | Qwen3.5-9B text encoder |
| `te_nvfp4` | `text_encoders/PAN2_qwen3_5_nvfp4.safetensors` | 4.6 GiB | Qwen3.5-9B text encoder |
| `vae_fp16` | `vae/PAN2_vae_fp16.safetensors` | 1.3 GiB | the fp32 VAE cast to fp16 |

Weight error against the original (relative, measured per layer): int8 ConvRot about 0.9%, NVFP4 about 9%.

## Requirements

- Python 3.10+ with `torch` (2.1 or newer, CPU build is enough), `safetensors`, `numpy`, `huggingface_hub`
- Disk: the download (transformer 63 GiB, text encoder 15 GiB, VAE 2.7 GiB) plus the outputs you choose
- RAM: 16 GiB or more recommended. No GPU needed.

```bash
pip install torch safetensors numpy huggingface_hub
```

## Use

Make everything, downloading what is needed into the Hugging Face cache:

```bash
python convert_pan2.py --out /path/to/ComfyUI/models
```

Only some files:

```bash
python convert_pan2.py --out /path/to/ComfyUI/models --only transformer_nvfp4 te_nvfp4 vae_fp16
```

From a copy you already downloaded (a folder containing `transformer/`, `text_encoder/`, `vae/`):

```bash
python convert_pan2.py --out /path/to/ComfyUI/models --src /path/to/PAN2
```

Files that already exist are skipped; add `--overwrite` to redo them. Each transformer file takes roughly ten
minutes on a desktop CPU.

`--device cuda` does the math on the GPU. It is faster, but a few values can round differently, so the result may
not be byte-identical to a CPU run.

## Check a result

```bash
python verify_pan2.py /path/to/PAN2/transformer /path/to/ComfyUI/models/diffusion_models/PAN2_nvfp4.safetensors
```

It confirms that every tensor is present, that everything not quantized is bit-identical to the source, and prints
the error of the quantized layers.

## Using the files

These load in a ComfyUI build that has PAN 2 support (model code, the `pan2` text encoder type, and loading of
diffusers-layout Wan VAEs). Stock ComfyUI without that support will not recognise them. One such build is
<https://github.com/speedyrulz/ComfyUI>.

RulzTrainer reads the same files to train PAN 2 LoRAs (`model_source: files`; see "PAN 2" in the trainer's README).

- Diffusion model: Load Diffusion Model
- Text encoder: Load CLIP, type `pan2`
- VAE: Load VAE

## What the script does

- Quantizes the large 2-D weights of the transformer blocks and the text refiner (for the text encoder: of its
  layers) and copies every other tensor unchanged.
- int8 ConvRot: each weight is rotated by a block Hadamard matrix of size 256, then stored as int8 with one scale
  per output row.
- NVFP4: 4-bit e2m1 values with one fp8 scale per block of 16 and one fp32 scale per tensor.
- NVFP4 text encoder: layers are marked `full_precision_matrix_mult`, and the token embedding table is stored as
  plain row-wise int8.
- The model's `config.json` is kept in the file's metadata.

PAN 2 itself is released by IFM under its own license; check the model page before redistributing converted files.
