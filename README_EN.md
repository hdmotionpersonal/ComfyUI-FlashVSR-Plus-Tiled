# FlashVSR+ T8 Spatial Tiling for ComfyUI

Native ComfyUI VIDEO and IMAGE super-resolution powered by [FlashVSR+](https://github.com/OpenImagingLab/FlashVSR_Plus). This fork adds spatial tiling with overlap blending for larger output frames, alongside audio preservation, incremental video processing, and INT8 ConvRot support. Requires an NVIDIA CUDA GPU.

[Models](https://huggingface.co/t8star/Flashvsr-Plus-Comfy) · [Example workflows](example_workflows) · [Validation](docs/VALIDATION.md) · [Spatial tiling PRD](PRD_SPATIAL_TILING_RU.md)

## Installation

In **ComfyUI Manager**, search for **FlashVSR+ Tiled**, install the node pack, and restart ComfyUI. The node category and searchable node names include `Tiled` to distinguish this fork from the original nodes.

To install manually, run from the ComfyUI directory and use **ComfyUI's own Python** to install dependencies:

```sh
git clone --branch spatial-tiling https://github.com/hdmotionpersonal/ComfyUI-FlashVSR-Plus-Tiled.git custom_nodes/ComfyUI-FlashVSR-Plus-Tiled
python -m pip install -r custom_nodes/ComfyUI-FlashVSR-Plus-Tiled/requirements.txt
```

The dependency list does not include PyTorch or Transformers and will not replace them. `transformers==4.57.1` was verified in an isolated environment. On Windows, Triton must match the installed PyTorch version; the verified combination is PyTorch `2.7.0+cu128` with `triton-windows==3.3.1.post21`.

## Quick start

1. Import the [VIDEO workflow](example_workflows/FlashVSR_Plus_Video.json) or [video-frame IMAGE workflow](example_workflows/FlashVSR_Plus_Frames.json). Copy the [sample video](example_workflows/flashvsr_plus_demo.mp4) into `ComfyUI/input`. For a still photo, use the [single-image workflow](example_workflows/FlashVSR_Plus_Single_Image.json).
2. In **FlashVSR+ Tiled Model Loader**, choose a mode, precision, attention backend, and decoder. For a first run, use `turbo`, `int8_convrot`, and `tiny`; leave `auto_download` enabled if the model files are not already installed. INT8 requires Comfy Kitchen `>=0.2.35`.
3. Set the scale or target dimensions and queue the workflow. The VIDEO path preserves audio, frame rate, and upstream trim/crop settings. Use VIDEO for longer clips.

| Node | Purpose |
| --- | --- |
| FlashVSR+ Tiled Model Loader | Load Turbo, Base, or V1 weights in BF16, FP16, or INT8 ConvRot with Tiny or Wan VAE decoding |
| FlashVSR+ Tiled Upscale Video | Upscale native VIDEO input with incremental reading and encoding |
| FlashVSR+ Tiled Upscale Frames | Upscale an RGB image or a batch of video frames and return run information |
| FlashVSR+ Tiled Options | Configure spatial tiles, overlap blending, temporal processing, cache offloading, decoding, and color correction |

## Spatial tiling

Connect **FlashVSR+ Tiled Options** to either tiled upscale node and enable `spatial_tiling`. The tile size and overlap are measured in output-canvas pixels. The canvas is padded to the model's 128-pixel alignment, tiled with the requested overlap, then cropped back to the requested output dimensions.

Choose one of three overlap blend methods:

- `cosine` — smooth cosine feathering across shared edges; recommended starting point.
- `linear` — linear feathering across shared edges.
- `gaussian` — center-weighted feathering with adjustable `spatial_gaussian_sigma`.

Before tiled inference, the ComfyUI console reports the tile size, final output dimensions, and tile grid (columns × rows). Weights from overlapping tiles are normalized by their sum. Canvas edges retain full weight where there is no neighboring tile. The console shows a separate `tqdm` progress bar for each tile, labeled with its size and canvas position.

| Option | Default | Notes |
| --- | ---: | --- |
| `spatial_tiling` | `false` | Enable spatial tile inference. |
| `spatial_tile_height` | `512` | Tile height in output pixels; must be a multiple of 128. |
| `spatial_tile_width` | `0` | Tile width in output pixels; `0` follows the height, otherwise set a multiple of 128. |
| `spatial_overlap` | `128` | Shared output pixels on both axes; must be a multiple of 128 and less than half of both tile dimensions. |
| `spatial_blend` | `cosine` | `cosine`, `linear`, or `gaussian`. |
| `spatial_gaussian_sigma` | `0.25` | Controls the Gaussian blend profile. |

Spatial VIDEO tiling currently requires the **Tiny** decoder and a positive numeric `decode_every` value. It writes temporary float16 tile segments to disk; temporary storage grows with clip length and the total tile area. Spatial VIDEO tiling with Wan VAE is not supported yet. Each tile uses its crop from the global latent-noise coordinates. Tile boundaries can still differ because each tile has a smaller attention context. Color correction in spatial VIDEO mode is applied per tile; disable it if you see color shifts at seams.

A GPU smoke test on an RTX 4070 Ti SUPER used V1 INT8 ConvRot, Tiny, SDPA, 512-pixel tiles, and 128-pixel overlap. The 33-frame demo clip completed at its full duration with cosine blending. Cosine, linear, and Gaussian each completed an 8-frame GPU run. The full-resolution output was 600×336; the padded canvas produced two horizontal tiles with exactly 128 pixels of overlap.

## Single-image upscaling

Import the [single-image workflow](example_workflows/FlashVSR_Plus_Single_Image.json), choose a static PNG or JPEG in **Load Image**, and run it to save a PNG through **Save Image**. Copy the [sample image](example_workflows/flashvsr_plus_sample.png) into `ComfyUI/input` to reproduce the example, or load the [tested output PNG with its embedded workflow](assets/single_image/portrait.png). See the [API workflow](example_workflows/FlashVSR_Plus_Single_Image_API.json) for programmatic use.

Defaults are `turbo / int8_convrot / tiny`, 4× scaling, fixed seed 1, and color correction enabled. BF16 and INT8 GPU checks covered a portrait, a 1080p photo, and a text chart. Photo enhancement passed on these samples; small text and flat graphics are not guaranteed to remain exact. Process unrelated images individually and preserve alpha separately. See [comparisons and acceptance scope](docs/SINGLE_IMAGE.md).

## Models and acceleration

Models are discovered in `ComfyUI/models/flashvsr_plus` and the node's `ckpts` directory. You can also set a custom model root in the loader. See the [model repository](https://huggingface.co/t8star/Flashvsr-Plus-Comfy) for the complete file layout. Downloads use pinned revisions and SHA-256 verification.

- **INT8 ConvRot:** each preconverted DiT is approximately **1.45 GB**. It runs with the matching LQ projection and decoder; the original DiT is not required. Transformer-block linear layers use Comfy Kitchen W8A8, while other components remain in floating point.
- **Attention:** official BSA, Triton, and SDPA backends are supported. `auto` selects an available backend; Triton is recommended for high-resolution output. SDPA rejects oversized masks before allocation. Native BSA has not been verified on Windows.
- **Segmented decoding:** Tiny accepts positive `decode_every` values divisible by four, such as `4`. Wan VAE uses full-clip decoding and requires `decode_every=0`.

INT8 primarily reduces weight memory; speed and image quality depend on the content and hardware. IMAGE output requires RAM for the full batch. VIDEO uses a constant-frame-rate timeline; normalize variable-frame-rate sources to CFR first. Audio that cannot be copied into MP4, such as PCM, is encoded as AAC. Model weights return to CPU after each request; `keep_loaded` retains the CPU copy between requests.

## Validation

**48 regression tests passed.** Spatial GPU checks covered all three blend methods, plus a full 33-frame cosine-blended video run on an RTX 4070 Ti SUPER. See [validation scope](docs/VALIDATION.md) and the [1080p validation clip](assets/validation_1080p_5s.mp4) for the existing full-resolution validation records.

## Community and resources

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)

## Credits and license

Based on [OpenImagingLab/FlashVSR_Plus](https://github.com/OpenImagingLab/FlashVSR_Plus) and [FlashVSR V1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1). INT8 linear operations use [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen). This ComfyUI integration and its derived checkpoints are maintained by T8star. Code and model files are licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
