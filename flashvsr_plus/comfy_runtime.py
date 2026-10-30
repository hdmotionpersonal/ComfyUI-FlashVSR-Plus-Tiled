"""Host-neutral service used by the ComfyUI nodes and integration tests."""
from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import random
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .backends import diagnostics, resolve_backend, validate_attention_size
from .inference import InferenceRequest, build_inference_pipeline
from .inference.presets import stream_config_for_mode
from .inference.postprocessing import WaveletColorCorrector
from .inference.preprocessing.input import _resize_tensor_video
from .inference.tiling import blend_weights, plan_tiles
from .quantization import convert_checkpoint, read_quantization
from .weights import ROOT, FILES, model_paths

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Options:
    spatial_tiling: bool = False
    spatial_tile_height: int = 512
    spatial_overlap: int = 128
    spatial_blend: str = "cosine"
    spatial_gaussian_sigma: float = 0.25
    temporal_compression: int = 4
    temporal_merge: str = "gaussian"
    temporal_gaussian_sigma: float = 1.0
    decode_every: int = 28
    cache_offload_mode: str = "auto"
    offload_original_cache: bool = False
    tiny_decoder_mode: str = "auto"
    tiny_decoder_channels_last: bool = True
    color_correction: bool = False
    color_correction_levels: int = 5
    color_correction_chunk_size: int = 16
    ffn_chunk_tokens: int = 8192
    vae_tiled: bool = True
    vae_tile_height: int = 30
    vae_tile_width: int = 52
    vae_overlap: int = 15
    video_preset: str = "veryfast"
    spatial_tile_width: int = 0

    def validate(self):
        if self.temporal_compression not in (1, 2, 4):
            raise ValueError("temporal_compression must be 1, 2 or 4")
        if self.decode_every < 0 or self.decode_every % 4:
            raise ValueError("decode_every must be 0 (auto) or a positive multiple of 4")
        if self.temporal_merge not in ("gaussian", "first") or not math.isfinite(self.temporal_gaussian_sigma) or self.temporal_gaussian_sigma <= 0:
            raise ValueError("Invalid temporal merge configuration")
        if self.cache_offload_mode not in ("auto", "never", "always"):
            raise ValueError("Invalid cache offload mode")
        if self.tiny_decoder_mode not in ("auto", "serial", "parallel"):
            raise ValueError("Invalid Tiny decoder mode")
        if min(self.color_correction_levels, self.color_correction_chunk_size,
               self.vae_tile_height, self.vae_tile_width) <= 0 or self.ffn_chunk_tokens < 0:
            raise ValueError("Invalid decoder/color/FFN configuration")
        if self.vae_overlap < 0 or self.vae_overlap >= min(self.vae_tile_height, self.vae_tile_width):
            raise ValueError("VAE overlap must be smaller than each tile dimension")
        if self.video_preset not in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"):
            raise ValueError("Invalid x264 encoding preset")
        if type(self.spatial_tiling) is not bool:
            raise TypeError("spatial_tiling must be a boolean")
        if type(self.spatial_tile_height) is not int or self.spatial_tile_height < 256 or self.spatial_tile_height % 128:
            raise ValueError("spatial_tile_height must be an integer >=256 and divisible by 128")
        if type(self.spatial_tile_width) is not int or (self.spatial_tile_width != 0 and
                (self.spatial_tile_width < 256 or self.spatial_tile_width % 128)):
            raise ValueError("spatial_tile_width must be 0 (same as height) or an integer >=256 divisible by 128")
        if type(self.spatial_overlap) is not int or self.spatial_overlap < 0 or self.spatial_overlap >= min(self.spatial_tile_height, self.spatial_tile_width or self.spatial_tile_height) / 2 or self.spatial_overlap % 128:
            raise ValueError("spatial_overlap must be nonnegative, divisible by 128, and smaller than half of both tile dimensions")
        if self.spatial_blend not in ("cosine", "linear", "gaussian"):
            raise ValueError("spatial_blend must be cosine, linear or gaussian")
        if not math.isfinite(self.spatial_gaussian_sigma) or not 0.05 <= self.spatial_gaussian_sigma <= 1.0:
            raise ValueError("spatial_gaussian_sigma must be finite and between 0.05 and 1.0")
        return self

    @property
    def effective_spatial_tile_width(self):
        return self.spatial_tile_width or self.spatial_tile_height


def geometry(source_width, source_height, scale, width=0, height=0):
    if min(source_width, source_height) <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Image dimensions and scale must be positive")
    if width < 0 or height < 0:
        raise ValueError("Target dimensions cannot be negative")
    if width and not height:
        height = max(1, round(source_height*width/source_width))
    elif height and not width:
        width = max(1, round(source_width*height/source_height))
    elif not width and not height:
        width, height = max(1, round(source_width*scale)), max(1, round(source_height*scale))
    return int(width), int(height), math.ceil(width/128)*128, math.ceil(height/128)*128


def report_spatial_plan(tiles, tile_width, tile_height, output_width, output_height,
                        progress=None):
    columns = len({tile.x for tile in tiles})
    rows = len({tile.y for tile in tiles})
    message = (f"Spatial plan: tile={tile_width}x{tile_height} px (W x H); "
               f"output={output_width}x{output_height} px (W x H); "
               f"grid={columns} tiles horizontally x {rows} vertically "
               f"({len(tiles)} total)")
    if progress is None:
        log.info(message)
    else:
        progress(message)
    return message


def image_to_model(images: torch.Tensor, width: int, height: int,
                   canvas_width: int, canvas_height: int) -> torch.Tensor:
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] < 1:
        raise ValueError("IMAGE must be a nonempty [frames,height,width,3] RGB tensor")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("IMAGE values must be finite and within [0,1]")
    # Keep the full clip in CPU memory; only bounded runner windows reach CUDA.
    value = images.detach().to(device="cpu", dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0)
    value = _resize_tensor_video(value, height, width)[0].permute(1, 0, 2, 3)
    value = F.pad(value, (0, canvas_width-width, 0, canvas_height-height), mode="replicate")
    return value.mul_(2).sub_(1).permute(1, 0, 2, 3).unsqueeze(0).contiguous()


@contextlib.contextmanager
def preserve_rng(device):
    py, np_state = random.getstate(), np.random.get_state()
    index = torch.device(device).index
    devices = list(range(torch.cuda.device_count()))
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


class ModelHandle:
    def __init__(self, *, mode="turbo", precision="bf16", attention="auto", decoder="tiny",
                 directory=None, auto_download=True, keep_loaded=False, device="cuda:0"):
        if mode not in ("turbo", "base", "v1") or decoder not in ("tiny", "vae"):
            raise ValueError("Invalid official model preset or decoder")
        if precision not in ("bf16", "fp16", "int8_convrot"):
            raise ValueError("Precision must be bf16, fp16 or int8_convrot")
        self.mode, self.precision, self.attention, self.decoder = mode, precision, attention, decoder
        self.directory = Path(directory or ROOT/"ckpts").expanduser().resolve()
        self.auto_download, self.keep_loaded, self.device = auto_download, keep_loaded, device
        self._lock, self._pipeline, self._key = threading.RLock(), None, None

    def description(self):
        return {"mode": self.mode, "precision": self.precision, "decoder": self.decoder,
                "attention": self.attention, "model_directory": str(self.directory),
                "device": self.device, "backends": diagnostics(self.device)}

    def release(self):
        if self._pipeline is not None:
            bundle = self._pipeline.bundle
            bundle.denoiser.interrupt_callback = None
            bundle.decoder.interrupt_callback = None
            bundle.denoiser.to("cpu")
            bundle.decoder.module.to("cpu")
            self._pipeline = None
            self._key = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build(self, config, progress):
        target = self.directory/self.mode/"dit_int8_convrot.safetensors"
        paths = model_paths(self.mode, self.directory, self.decoder,
                            auto_download=self.auto_download, progress=progress,
                            include_dit=(self.precision != "int8_convrot" or (not target.is_file() and not self.auto_download)),
                            include_int8=(self.precision == "int8_convrot" and not target.is_file() and self.auto_download))
        if self.precision == "int8_convrot":
            if not target.exists():
                convert_checkpoint(paths.dit, target, mode=self.mode, device=self.device, progress=progress)
            spec = read_quantization(target)
            if spec is None or spec["mode"] != self.mode:
                raise ValueError("INT8 checkpoint mode/format does not match the requested model")
            expected_source = next(x[-1] for x in FILES if x[0] == self.mode + "_dit")
            if spec["source_sha256"] != expected_source:
                raise ValueError("INT8 checkpoint was not converted from the expected official preset")
            paths.dit = str(target)
        progress("Loading FlashVSR+ weights")
        return build_inference_pipeline(config, paths)

    def infer(self, request: InferenceRequest, *, seed=1, options: Options | None = None,
              progress: Callable[[str], None] | None = None, interrupt: Callable[[], None] | None = None,
              segment_writer=None):
        options = (options or Options(decode_every=(0 if self.decoder == "vae" else 28))).validate()
        progress = progress or (lambda message: log.info("FlashVSR+: %s", message))
        interrupt = interrupt or (lambda: None)
        with self._lock:
            interrupt()
            request.validate()
            backend = resolve_backend(self.attention, self.device)
            h, w = (((request.spatial_crop_height, request.spatial_crop_width)
                     if request.spatial_crop_height is not None else (request.height, request.width))
                    if request.height is not None else
                    request.lq_video.shape[-2:] if request.lq_video is not None else (request.height, request.width))
            validate_attention_size(h, w, backend)
            if self.decoder == "vae" and request.input_path is not None and options.decode_every:
                raise ValueError("Official VAE decoding requires decode_every=0 (auto); use Tiny for bounded long-video decoding")
            decode_every = options.decode_every if self.decoder == "tiny" and options.decode_every else "auto"
            config = stream_config_for_mode(self.mode, device=self.device, seed=int(seed),
                decode_every=decode_every, tiny_decoder_mode=options.tiny_decoder_mode,
                tiny_decoder_channels_last=options.tiny_decoder_channels_last,
                offload_original_cache=options.offload_original_cache, cache_offload_mode=options.cache_offload_mode,
                temporal_compression=options.temporal_compression, temporal_merge=options.temporal_merge,
                temporal_gaussian_sigma=options.temporal_gaussian_sigma, attn_kernel=backend,
                color_correction=options.color_correction, color_correction_levels=options.color_correction_levels,
                color_correction_chunk_size=options.color_correction_chunk_size)
            config.spatial_noise_origin = (
                ((request.noise_origin_y if request.noise_origin_y is not None else request.spatial_crop_y),
                 (request.noise_origin_x if request.noise_origin_x is not None else request.spatial_crop_x))
                if request.noise_origin_y is not None or request.spatial_crop_y is not None else None
            )
            config.spatial_noise_canvas = (
                ((request.noise_canvas_height if request.noise_canvas_height is not None else request.height),
                 (request.noise_canvas_width if request.noise_canvas_width is not None else request.width))
                if request.noise_canvas_height is not None or request.spatial_crop_y is not None else None
            )
            config.dtype = "fp16" if self.precision == "fp16" else "bf16"
            config.stream_ffn_chunk_tokens = options.ffn_chunk_tokens or None
            config.decoder.kind, config.decoder.tiled = self.decoder, options.vae_tiled
            config.decoder.tile_size = (options.vae_tile_height, options.vae_tile_width)
            config.decoder.tile_stride = (options.vae_tile_height-options.vae_overlap, options.vae_tile_width-options.vae_overlap)
            config.validate()
            key = dataclasses.asdict(config)
            key.pop("seed")
            key.pop("spatial_noise_origin")
            key.pop("spatial_noise_canvas")
            try:
                # Cooperate with the host before loading our private pipeline.
                try:
                    from comfy import model_management
                except ImportError:
                    pass
                else:
                    needed = (3.8 if self.precision != "int8_convrot" else 2.5)*1024**3 + h*w*800
                    model_management.free_memory(needed, torch.device(self.device))
                with preserve_rng(self.device), torch.inference_mode():
                    if self._pipeline is None or self._key != key:
                        self.release()
                        self._pipeline = self._build(config, progress)
                        self._key = key
                    else:
                        self._pipeline.config.seed = int(seed)
                        self._pipeline.bundle.denoiser.to(self.device)
                        self._pipeline.bundle.decoder.module.to(self.device)
                    self._pipeline.bundle.denoiser.interrupt_callback = interrupt
                    self._pipeline.bundle.decoder.interrupt_callback = interrupt
                    def callback(message):
                        interrupt()
                        progress(message)
                    result = self._pipeline.infer(request, progress=callback,
                        video_preset=options.video_preset, segment_writer=segment_writer)
                    result.metadata.update({"mode": self.mode, "precision": self.precision,
                                            "actual_backend": backend, "seed": int(seed)})
                    return result
            finally:
                if self._pipeline is not None:
                    self._pipeline.bundle.denoiser.interrupt_callback = None
                    self._pipeline.bundle.decoder.interrupt_callback = None
                    self._pipeline.bundle.denoiser.to("cpu")
                    self._pipeline.bundle.decoder.module.to("cpu")
                if not self.keep_loaded:
                    self.release()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def upscale_images(model: ModelHandle, images, *, scale=4.0, width=0, height=0,
                   seed=1, fps=24.0, options=None, progress=None, interrupt=None,
                   tile_progress_factory=None):
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("Expected a ComfyUI IMAGE batch")
    width, height, cw, ch = geometry(images.shape[2], images.shape[1], scale, width, height)
    options = (options or Options()).validate()
    backend = resolve_backend(model.attention, model.device)
    if options.spatial_tiling:
        tile_width = options.effective_spatial_tile_width
        tiles = plan_tiles(ch, cw, options.spatial_tile_height, options.spatial_overlap,
                           tile_width=tile_width)
        report_spatial_plan(tiles, tile_width, options.spatial_tile_height,
                            width, height, progress)
        validate_attention_size(min(ch, options.spatial_tile_height),
                                min(cw, tile_width), backend)
    else:
        tiles = ()
        validate_attention_size(ch, cw, backend)
    video = image_to_model(images, width, height, cw, ch)
    if not options.spatial_tiling or len(tiles) == 1:
        request = InferenceRequest(lq_video=video, fps=float(fps))
        result = model.infer(request, seed=seed, options=options, progress=progress, interrupt=interrupt)
        stitched = result.video[0].float()
    else:
        accum = torch.zeros((3, video.shape[2], ch, cw), dtype=torch.float32)
        weights = torch.zeros((1, 1, ch, cw), dtype=torch.float32)
        was_keep_loaded = model.keep_loaded
        model.keep_loaded = True
        result = None
        try:
            for index, tile in enumerate(tiles):
                if interrupt is not None:
                    interrupt()
                tile_video = video[..., tile.y:tile.y + tile.height, tile.x:tile.x + tile.width]
                expected_latents = (max(0, video.shape[2] - 1) // 4) + 1
                expected_chunks = max(1, (expected_latents + 3) // 4)
                tile_context = (tile_progress_factory(index, len(tiles), tile, expected_chunks)
                                if tile_progress_factory is not None else contextlib.nullcontext(progress))
                with tile_context as tile_callback:
                    if tile_progress_factory is None and progress is not None:
                        progress(f"Spatial tile {index + 1}/{len(tiles)}")
                    tile_result = model.infer(
                        InferenceRequest(lq_video=tile_video, fps=float(fps),
                                         noise_origin_y=tile.y, noise_origin_x=tile.x,
                                         noise_canvas_height=ch, noise_canvas_width=cw), seed=seed,
                        options=dataclasses.replace(options, color_correction=False),
                        progress=tile_callback, interrupt=interrupt,
                    )
                prediction = tile_result.video[0].float()
                weight = blend_weights(tile, ch, cw, tiles, options.spatial_blend,
                                       options.spatial_gaussian_sigma)
                accum[..., tile.y:tile.y + tile.height, tile.x:tile.x + tile.width] += prediction * weight
                weights[..., tile.y:tile.y + tile.height, tile.x:tile.x + tile.width] += weight
                result = tile_result
                del prediction, tile_video, tile_result
        finally:
            model.keep_loaded = was_keep_loaded
            if not was_keep_loaded:
                model.release()
        if torch.any(weights <= 0):
            raise RuntimeError("Spatial tile plan left uncovered pixels")
        stitched = accum / weights
        if options.color_correction:
            corrector = WaveletColorCorrector(
                levels=options.color_correction_levels,
                chunk_size=options.color_correction_chunk_size,
            )
            stitched = corrector(stitched.unsqueeze(0), video[:, :, :stitched.shape[1]])[0]
        result.metadata.update({"spatial_tiling": True, "tile_count": len(tiles),
                                "tile_height": options.spatial_tile_height,
                                "tile_width": tile_width,
                                "overlap": options.spatial_overlap,
                                "blend": options.spatial_blend,
                                "gaussian_sigma": options.spatial_gaussian_sigma,
                                "noise_policy": "global latent coordinates"})
    output = stitched[:, :, :height, :width].permute(1, 2, 3, 0).to(
        dtype=torch.float32, copy=True, memory_format=torch.contiguous_format).add_(1).div_(2).clamp_(0, 1)
    result.metadata.update({"width": width, "height": height, "canvas_width": cw, "canvas_height": ch})
    return output, result
