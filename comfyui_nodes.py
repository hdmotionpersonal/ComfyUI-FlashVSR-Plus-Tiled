from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import folder_paths
import comfy.model_management as mm
import comfy.utils

from .flashvsr_plus.comfy_runtime import (ModelHandle, Options, geometry,
                                          report_spatial_plan, upscale_images)
from .flashvsr_plus.inference import InferenceRequest
from .flashvsr_plus.inference.output import Float16SegmentSpool, IncrementalVideoWriter, mux_source_audio
from .flashvsr_plus.inference.tiling import plan_tiles, stitch_tiles
from .flashvsr_plus.weights import ROOT, FILES

CATEGORY = "FlashVSR+ Tiled"
MODEL_TYPE = "FLASHVSR_PLUS_MODEL"
_OUTPUT_COMMIT_LOCK = threading.Lock()
OPTIONS_TYPE = "FLASHVSR_PLUS_OPTIONS"
MODEL_DIR = Path(folder_paths.models_dir)/"flashvsr_plus"
folder_paths.add_model_folder_path("flashvsr_plus", str(MODEL_DIR))
if (ROOT/"ckpts").is_dir():
    folder_paths.add_model_folder_path("flashvsr_plus", str(ROOT/"ckpts"))


def _progress():
    bar = comfy.utils.ProgressBar(100)
    chunks = 0
    def update(message, overall_fraction=None, *, quiet=False, log_writer=None):
        nonlocal chunks
        mm.throw_exception_if_processing_interrupted()
        if not quiet:
            formatted = f"[INFO] [FlashVSR+] {message}"
            if log_writer is None:
                logging.info("[FlashVSR+] %s", message)
            else:
                log_writer(formatted)
        if overall_fraction is not None:
            bar.update_absolute(round(95 * min(1.0, max(0.0, overall_fraction))), 100)
            return
        match = re.match(r"DiT chunk (\d+)/(\d+)", message)
        if match:
            bar.update_absolute(min(95, round(90*int(match[1])/int(match[2]))), 100)
        elif message.startswith("DiT chunk"):
            chunks += 1
            bar.update_absolute(min(90, chunks), 100)
        elif "complete" in message.lower() or "saved" in message.lower():
            bar.update_absolute(98, 100)
    return bar, update


@contextmanager
def _tile_progress(index, count, tile, progress, expected_chunks=None,
                   expected_decode_segments=None):
    label = f"Tile {index + 1}/{count} x={tile.x},y={tile.y}"
    # Tiles run sequentially, so one in-place console row per tile is enough.
    # A fixed compact format avoids terminal wrapping and cursor-up ANSI codes.
    bar = tqdm(total=100, desc=label, position=0, leave=True, dynamic_ncols=True,
               bar_format="{desc}: {percentage:3.0f}%|{bar:32}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
               mininterval=0.2)

    def write_log(message):
        tqdm.write(message, file=bar.fp)
    current = 0
    chunks = 0
    decode_segments = 0

    def combined_progress():
        chunk_fraction = min(1.0, chunks / expected_chunks) if expected_chunks else 0.0
        decode_fraction = (min(1.0, decode_segments / expected_decode_segments)
                           if expected_decode_segments else 0.0)
        return round(80 * chunk_fraction + 20 * decode_fraction)

    def update(message):
        nonlocal current, chunks, decode_segments
        match = re.match(r"DiT chunk (\d+)/(\d+|\?)", message)
        decode_match = re.match(r"Decode segment (\d+)(?:/(\d+))?", message)
        if match:
            chunks = max(chunks, int(match[1]))
            total_chunks = int(match[2]) if match[2].isdigit() else expected_chunks
            value = combined_progress() if total_chunks else min(80, chunks)
        elif decode_match:
            decode_segments = max(decode_segments, int(decode_match[1]))
            total_segments = (int(decode_match[2]) if decode_match[2]
                              else expected_decode_segments)
            value = combined_progress() if total_segments else min(98, current + 1)
        elif message.startswith("DiT chunk"):
            chunks += 1
            value = combined_progress() if expected_chunks else min(80, chunks)
        elif "complete" in message.lower() or "saved" in message.lower():
            value = 99
        else:
            if progress is not None:
                progress(f"Spatial {label}: {message}", log_writer=write_log)
            return
        value = max(current, value)
        if value > current:
            bar.update(value - current)
            current = value
        if progress is not None:
            progress(f"Spatial {label}: {message}",
                     overall_fraction=(index + current / 100) / count,
                     quiet=message.startswith("DiT chunk"),
                     log_writer=write_log)

    try:
        if progress is not None:
            progress(f"Spatial {label}", log_writer=write_log)
        yield update
    except BaseException:
        bar.set_description(f"{label} failed")
        raise
    else:
        if current < 100:
            bar.update(100 - current)
        if progress is not None:
            progress(f"Spatial {label} complete", overall_fraction=(index + 1) / count,
                     log_writer=write_log)
    finally:
        bar.close()


def _output_is_connected(prompt, node_id, output_index):
    if not isinstance(prompt, dict) or node_id is None:
        return True
    node_id = str(node_id)
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        for link in (node.get("inputs") or {}).values():
            if (isinstance(link, (list, tuple)) and len(link) == 2
                    and str(link[0]) == node_id and link[1] == output_index):
                return True
    return False


def _extract_audio_track(path):
    import av

    with av.open(str(path), mode="r") as container:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            return None
        resampler = av.AudioResampler(format="fltp")
        chunks = []
        for packet in container.demux(stream):
            for decoded in packet.decode():
                chunks.extend(frame.to_ndarray() for frame in resampler.resample(decoded))
        chunks.extend(frame.to_ndarray() for frame in resampler.resample(None))
        if not chunks:
            return None
        waveform = torch.from_numpy(np.concatenate(chunks, axis=1)).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": int(stream.sample_rate or 1)}


class FlashVSRPlusModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (["turbo", "base", "v1"],),
            "precision": (["bf16", "int8_convrot", "fp16"],),
            "attention": (["auto", "triton_block_sparse", "block_sparse", "sdpa"],),
            "decoder": (["tiny", "vae"],),
            "auto_download": ("BOOLEAN", {"default": True}),
            "keep_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep weights in CPU RAM between requests. GPU memory is released after each request."}),
            "model_directory": ("STRING", {"default": "", "tooltip": "Optional model root; blank discovers models/flashvsr_plus and the bundled ckpts folder."}),
        }}
    RETURN_TYPES = (MODEL_TYPE, "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, mode="turbo", precision="bf16", attention="auto", decoder="tiny",
             auto_download=True, keep_loaded=False, model_directory=""):
        device = mm.get_torch_device()
        if device.type != "cuda":
            raise RuntimeError("Official FlashVSR+ requires an NVIDIA CUDA GPU")
        directories = [Path(x) for x in folder_paths.get_folder_paths("flashvsr_plus")]
        if model_directory.strip():
            directory = Path(model_directory).expanduser()
        else:
            weight_names = (("dit_int8_convrot.safetensors", "dit.safetensors")
                            if precision == "int8_convrot" else ("dit.safetensors",))
            candidates = [d for name in weight_names for d in directories if (d/mode/name).is_file()]
            components = [entry[1] for entry in FILES if entry[0] in {mode+"_lq", decoder}]
            directory = next((d for d in candidates if all((d/name).is_file() for name in components)),
                             candidates[0] if candidates else MODEL_DIR)
        handle = ModelHandle(mode=mode, precision=precision, attention=attention, decoder=decoder,
            directory=directory, auto_download=auto_download, keep_loaded=keep_loaded, device=str(device))
        return handle, json.dumps(handle.description(), ensure_ascii=False, indent=2)


class FlashVSRPlusOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "spatial_tiling": ("BOOLEAN", {"default": False, "tooltip": "Run Frames or Tiny-decoder VIDEO inference on overlapping spatial tiles. VIDEO stores temporary float32 segments on disk."}),
            "spatial_tile_height": ("INT", {"default": 512, "min": 256, "max": 8192, "step": 128, "tooltip": "Tile height in output pixels."}),
            "spatial_tile_width": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 128, "tooltip": "Tile width in output pixels. Set independently from height; 0 follows tile height for older workflows."}),
            "spatial_overlap": ("INT", {"default": 128, "min": 0, "max": 4096, "step": 8, "tooltip": "Overlap in output pixels; step 8, less than half of both tile dimensions. Tile dimensions stay multiples of 128."}),
            "spatial_blend": (["cosine", "linear", "gaussian"],),
            "spatial_gaussian_sigma": ("FLOAT", {"default": .25, "min": .05, "max": 1., "step": .05}),
            "temporal_compression": ([4, 2, 1],),
            "temporal_merge": (["gaussian", "first"],),
            "temporal_gaussian_sigma": ("FLOAT", {"default": 1., "min": .01, "max": 10., "step": .05}),
            "decode_every_frames": ("INT", {"default": 112, "min": 0, "max": 16384, "step": 4, "tooltip": "Output video-frame interval for Tiny incremental decoding. Must be divisible by 4 × temporal_compression. 0=auto/full clip; spatial VIDEO tiling requires a positive interval."}),
            "cache_offload_mode": (["auto", "never", "always"],),
            "offload_original_cache": ("BOOLEAN", {"default": False}),
            "tiny_decoder_mode": (["auto", "serial", "parallel"],),
            "tiny_decoder_channels_last": ("BOOLEAN", {"default": True}),
            "color_correction": ("BOOLEAN", {"default": False}),
            "color_correction_levels": ("INT", {"default": 5, "min": 1, "max": 8}),
            "color_correction_chunk_size": ("INT", {"default": 16, "min": 1, "max": 128}),
            "ffn_chunk_tokens": ("INT", {"default": 8192, "min": 0, "max": 262144, "step": 1024}),
            "vae_tiled": ("BOOLEAN", {"default": True}),
            "vae_tile_height": ("INT", {"default": 240, "min": 32, "max": 2048, "step": 8, "tooltip": "VAE decoder tile height in output pixels (Wan VAE uses 8 latent cells per pixel block)."}),
            "vae_tile_width": ("INT", {"default": 416, "min": 32, "max": 2048, "step": 8, "tooltip": "VAE decoder tile width in output pixels (Wan VAE uses 8 latent cells per pixel block)."}),
            "vae_overlap": ("INT", {"default": 120, "min": 0, "max": 1024, "step": 8, "tooltip": "Overlap between VAE decoder tiles in output pixels."}),
            "video_preset": (["veryfast", "ultrafast", "superfast", "faster", "fast", "medium", "slow"],),
        }}
    RETURN_TYPES = (OPTIONS_TYPE,)
    RETURN_NAMES = ("options",)
    FUNCTION = "configure"
    CATEGORY = CATEGORY

    def configure(self, **kwargs):
        # New UI values are in output pixels/frames. Keep saved workflows made
        # with the previous latent-frame controls working when their old values
        # do not align to the new pixel/frame increments.
        temporal_compression = kwargs["temporal_compression"]
        decode_every = kwargs.pop("decode_every_frames", kwargs.pop("decode_every", 112))
        frame_step = 4 * temporal_compression
        if decode_every and decode_every % frame_step:
            decode_every *= temporal_compression  # legacy latent-frame value
        kwargs["decode_every"] = decode_every

        vae_values = (kwargs["vae_tile_height"], kwargs["vae_tile_width"], kwargs["vae_overlap"])
        if all(value % 8 == 0 for value in vae_values):
            # Wan's VAE uses 8x spatial downsampling: convert displayed pixels
            # to the latent-grid dimensions expected by the decoder.
            kwargs["vae_tile_height"] //= 8
            kwargs["vae_tile_width"] //= 8
            kwargs["vae_overlap"] //= 8
        return (Options(**kwargs).validate(),)


class FlashVSRPlusUpscaleFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (MODEL_TYPE,), "images": ("IMAGE",),
            "scale": ("FLOAT", {"default": 4., "min": .25, "max": 16., "step": .25}),
            "width": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "0 uses scale or preserves aspect ratio from height."}),
            "height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
        }, "optional": {"options": (OPTIONS_TYPE,)}}
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    def upscale(self, model, images, scale=4., width=0, height=0, seed=1, options=None):
        bar, progress = _progress()
        output, result = upscale_images(model, images, scale=scale, width=width, height=height,
            seed=seed, options=options, progress=progress,
            interrupt=mm.throw_exception_if_processing_interrupted,
            tile_progress_factory=lambda i, n, tile, chunks: _tile_progress(
                i, n, tile, progress, expected_chunks=chunks))
        bar.update_absolute(100, 100)
        return output, json.dumps(result.metadata, ensure_ascii=False, indent=2)


class FlashVSRPlusUpscaleVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (MODEL_TYPE,), "video": ("VIDEO",),
            "scale": ("FLOAT", {"default": 4., "min": .25, "max": 16., "step": .25}),
            "width": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "max_frames": ("INT", {"default": 0, "min": 0, "max": 10000000}),
            "filename_prefix": ("STRING", {"default": "FlashVSRPlus/FlashVSRPlus"}),
        }, "optional": {"options": (OPTIONS_TYPE,)},
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}
    # Preserve the original output indices for saved workflows; append the
    # component outputs so existing video/info connections remain valid.
    RETURN_TYPES = ("VIDEO", "STRING", "IMAGE", "AUDIO")
    RETURN_NAMES = ("video", "info", "images", "audio")
    OUTPUT_TOOLTIPS = (
        "The upscaled video file.",
        "JSON metadata for the upscaled video.",
        "All upscaled frames as an IMAGE batch. Connecting this output loads the full batch into host RAM.",
        "The audio waveform preserved from the input video, or None when no audio track exists.",
    )
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    def upscale(self, model, video, scale=4., width=0, height=0, seed=1,
                max_frames=0, filename_prefix="FlashVSRPlus/FlashVSRPlus", options=None,
                prompt=None, unique_id=None):
        from comfy_api.input_impl import VideoFromFile
        from comfy_api.latest import Types
        options = options or Options(decode_every=(0 if model.decoder == "vae" else 112))
        if model.decoder == "vae" and options.decode_every:
            raise ValueError("VAE uses full-clip decoding: set decode_every=0 in FlashVSR+ Tiled Options. Tiny supports incremental long videos.")
        sw, sh = video.get_dimensions()
        width, height, cw, ch = geometry(sw, sh, scale, width, height)
        folder, name, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height)
        destination = Path(folder)/f"{name}_{counter:05}_.mp4"
        Path(folder).mkdir(parents=True, exist_ok=True)
        temp_root = Path(folder_paths.get_temp_directory())
        temp_root.mkdir(parents=True, exist_ok=True)
        bar, progress = _progress()
        with tempfile.TemporaryDirectory(prefix="flashvsr_plus_", dir=temp_root) as temporary, \
                tempfile.TemporaryDirectory(prefix=".flashvsr_plus_", dir=folder) as output_temporary:
            source, restored = Path(temporary)/"source.mp4", Path(output_temporary)/"restored.mp4"
            progress("Preparing video with its active trim, crop and audio")
            video.save_to(str(source), format=Types.VideoContainer.MP4,
                          codec=Types.VideoCodec.H264, crf=0, preset="ultrafast")
            mm.throw_exception_if_processing_interrupted()
            fps = float(video.get_frame_rate())
            if options.spatial_tiling:
                if model.decoder != "tiny":
                    raise ValueError("Spatial VIDEO tiling currently requires the Tiny decoder")
                if options.decode_every == 0:
                    raise ValueError("Spatial VIDEO tiling requires a numeric Tiny decode_every interval")
                tile_width = options.effective_spatial_tile_width
                tiles = plan_tiles(ch, cw, options.spatial_tile_height, options.spatial_overlap,
                                   tile_width=tile_width, pad_edges=True)
                cw = max(cw, max(tile.x + tile.width for tile in tiles))
                ch = max(ch, max(tile.y + tile.height for tile in tiles))
                report_spatial_plan(tiles, tile_width, options.spatial_tile_height,
                                    width, height, progress)
                if len(tiles) == 1:
                    result = model.infer(InferenceRequest(input_path=source, output_path=restored,
                        height=ch, width=cw, content_height=height, content_width=width,
                        max_frames=max_frames or None, fps=fps), seed=seed, options=options,
                        progress=progress, interrupt=mm.throw_exception_if_processing_interrupted)
                else:
                    spools, tile_results = [], []
                    was_keep_loaded = model.keep_loaded
                    model.keep_loaded = True
                    frame_count = video.get_frame_count()
                    if max_frames:
                        frame_count = min(frame_count, max_frames)
                    temporal_repeat = 4 // options.temporal_compression
                    expanded_frames = 1 + max(0, frame_count - 1) * temporal_repeat
                    latent_frames = (max(0, expanded_frames - 1) // 4) + 1
                    expected_chunks = max(1, (latent_frames + 3) // 4)
                    decode_latents = options.decode_every // options.temporal_compression
                    expected_decode_segments = max(1, (latent_frames + decode_latents - 1) // decode_latents)
                    try:
                        for index, tile in enumerate(tiles):
                            mm.throw_exception_if_processing_interrupted()
                            spool = Float16SegmentSpool(Path(temporary)/f"tile_{index:04d}")
                            request = InferenceRequest(input_path=source, height=ch, width=cw,
                                content_height=height, content_width=width,
                                max_frames=max_frames or None, fps=fps,
                                spatial_crop_y=tile.y, spatial_crop_x=tile.x,
                                spatial_crop_height=tile.height, spatial_crop_width=tile.width)
                            with _tile_progress(index, len(tiles), tile, progress,
                                                expected_chunks=expected_chunks,
                                                expected_decode_segments=expected_decode_segments) as tile_progress:
                                tile_result = model.infer(request, seed=seed, options=options,
                                    progress=tile_progress, interrupt=mm.throw_exception_if_processing_interrupted,
                                    segment_writer=spool)
                            spools.append(spool.verify())
                            tile_results.append(tile_result)
                    finally:
                        model.keep_loaded = was_keep_loaded
                        if not was_keep_loaded:
                            model.release()
                    segment_count = len(spools[0]["segments"])
                    if any(len(spool["segments"]) != segment_count for spool in spools):
                        raise RuntimeError("Spatial tiles produced different decode segment counts")
                    writer = IncrementalVideoWriter(restored, fps, preset=options.video_preset,
                                                    crop_hw=(height, width))
                    try:
                        for segment_index in range(segment_count):
                            mm.throw_exception_if_processing_interrupted()
                            predictions = [torch.from_numpy(np.load(
                                spool["segments"][segment_index]["path"], allow_pickle=False).astype(np.float32))
                                for spool in spools]
                            frame_counts = {value.shape[1] for value in predictions}
                            if len(frame_counts) != 1:
                                raise RuntimeError("Spatial tiles produced mismatched frame segments")
                            merged = stitch_tiles(ch, cw, tiles, predictions,
                                method=options.spatial_blend,
                                sigma_scale=options.spatial_gaussian_sigma)
                            writer.append(merged.unsqueeze(0))
                        writer.close()
                    except BaseException:
                        writer.close()
                        raise
                    artifact = writer.verify()
                    muxed = Path(output_temporary)/"restored_audio.mp4"
                    mux_source_audio(restored, source, muxed)
                    os.replace(muxed, restored)
                    result = tile_results[-1]
                    result.video = None
                    result.output_path = str(restored)
                    result.frames = artifact["frames"]
                    result.metadata.update({"artifact": artifact, "spatial_tiling": True,
                        "tile_count": len(tiles), "tile_height": options.spatial_tile_height,
                        "tile_width": tile_width,
                        "overlap": options.spatial_overlap, "blend": options.spatial_blend,
                        "gaussian_sigma": options.spatial_gaussian_sigma,
                        "temporary_spool_bytes": sum(spool["bytes"] for spool in spools),
                        "noise_policy": "global latent coordinates",
                        "color_correction_scope": "per tile" if options.color_correction else "disabled"})
            else:
                result = model.infer(InferenceRequest(input_path=source, output_path=restored,
                    height=ch, width=cw, content_height=height, content_width=width,
                    max_frames=max_frames or None, fps=fps), seed=seed, options=options, progress=progress,
                    interrupt=mm.throw_exception_if_processing_interrupted)
            # Committing only a complete file prevents cancelled jobs leaving published partial output.
            with _OUTPUT_COMMIT_LOCK:
                mm.throw_exception_if_processing_interrupted()
                while destination.exists():
                    counter += 1
                    destination = Path(folder)/f"{name}_{counter:05}_.mp4"
                os.replace(restored, destination)
        result.output_path = str(destination)
        if result.metadata.get("artifact"):
            result.metadata["artifact"]["path"] = str(destination)
        result.metadata.update({"path": str(destination), "width": width, "height": height,
                                "canvas_width": cw, "canvas_height": ch})
        bar.update_absolute(100, 100)
        output_video = VideoFromFile(str(destination))
        want_images = _output_is_connected(prompt, unique_id, 2)
        want_audio = _output_is_connected(prompt, unique_id, 3)
        image_output = audio_output = None
        if want_images:
            frame_count = output_video.get_frame_count()
            out_width, out_height = output_video.get_dimensions()
            estimated_gib = frame_count * out_width * out_height * 3 * 4 / (1024 ** 3)
            progress(f"Extracting {frame_count} output frames as IMAGE; estimated host RAM {estimated_gib:.2f} GiB")
            components = output_video.get_components()
            image_output = components.images
            if want_audio:
                audio_output = components.audio
        elif want_audio:
            progress("Extracting AUDIO waveform from completed video")
            audio_output = _extract_audio_track(destination)
        result.metadata.update({"image_output_connected": want_images,
                                "audio_output_connected": want_audio,
                                "audio_output_available": audio_output is not None})
        return (output_video, json.dumps(result.metadata, ensure_ascii=False, indent=2),
                image_output, audio_output)


NODE_CLASS_MAPPINGS = {
    "FlashVSRPlusTiledModelLoader": FlashVSRPlusModelLoader,
    "FlashVSRPlusTiledOptions": FlashVSRPlusOptions,
    "FlashVSRPlusTiledUpscaleFrames": FlashVSRPlusUpscaleFrames,
    "FlashVSRPlusTiledUpscaleVideo": FlashVSRPlusUpscaleVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashVSRPlusTiledModelLoader": "FlashVSR+ Tiled Model Loader",
    "FlashVSRPlusTiledOptions": "FlashVSR+ Tiled Options",
    "FlashVSRPlusTiledUpscaleFrames": "FlashVSR+ Tiled Upscale Frames",
    "FlashVSRPlusTiledUpscaleVideo": "FlashVSR+ Tiled Upscale Video",
}
