"""Spatial tile planning and normalized overlap blending for inference."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Tile:
    y: int
    x: int
    height: int
    width: int


def _starts(length: int, tile: int, overlap: int, alignment: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    positions = [0]
    while positions[-1] + tile < length:
        positions.append(positions[-1] + stride)
    return positions


def plan_tiles(height: int, width: int, tile_height: int, overlap: int, *,
               tile_width: int | None = None, alignment: int = 128) -> tuple[Tile, ...]:
    """Cover a canvas with edge-aligned tiles and no gaps."""
    tile_width = tile_height if tile_width is None or tile_width == 0 else tile_width
    if min(height, width, tile_height, tile_width, alignment) <= 0:
        raise ValueError("canvas, tile size and alignment must be positive")
    if height % alignment or width % alignment:
        raise ValueError("canvas dimensions must be divisible by alignment")
    if tile_height % alignment or tile_width % alignment:
        raise ValueError("tile dimensions must be divisible by alignment")
    if overlap % alignment:
        raise ValueError("overlap must be divisible by alignment")
    if overlap < 0 or overlap >= min(tile_height, tile_width) / 2:
        raise ValueError("overlap must be nonnegative and smaller than half the tile")
    ys = _starts(height, tile_height, overlap, alignment)
    xs = _starts(width, tile_width, overlap, alignment)
    return tuple(Tile(y, x, min(tile_height, height-y), min(tile_width, width-x))
                 for y in ys for x in xs)


def blend_weights(tile: Tile, canvas_height: int, canvas_width: int, tiles: tuple[Tile, ...],
                  method: str = "cosine", sigma_scale: float = 0.25, *, device="cpu") -> torch.Tensor:
    """Return one float32 2-D weight map; only shared edges are feathered."""
    if method not in {"linear", "cosine", "gaussian"}:
        raise ValueError("method must be linear, cosine or gaussian")
    if not math.isfinite(sigma_scale) or sigma_scale <= 0:
        raise ValueError("sigma_scale must be positive and finite")
    dtype = torch.float32
    weight = torch.ones((tile.height, tile.width), dtype=dtype, device=device)
    top = any(other.y < tile.y and other.y + other.height > tile.y for other in tiles)
    bottom = any(other.y > tile.y and other.y < tile.y + tile.height for other in tiles)
    left = any(other.x < tile.x and other.x + other.width > tile.x for other in tiles)
    right = any(other.x > tile.x and other.x < tile.x + tile.width for other in tiles)
    if method == "gaussian":
        y = torch.arange(tile.height, dtype=dtype, device=device)
        x = torch.arange(tile.width, dtype=dtype, device=device)
        gy = torch.exp(-0.5 * ((y - (tile.height - 1) / 2) / max(sigma_scale * tile.height, 1e-6)) ** 2)
        gx = torch.exp(-0.5 * ((x - (tile.width - 1) / 2) / max(sigma_scale * tile.width, 1e-6)) ** 2)
        weight = gy[:, None] * gx[None, :]
        # A true canvas edge has no competing tile and must retain full weight.
        if not top: weight[:max(1, tile.height // 8)] = 1
        if not bottom: weight[-max(1, tile.height // 8):] = 1
        if not left: weight[:, :max(1, tile.width // 8)] = 1
        if not right: weight[:, -max(1, tile.width // 8):] = 1
        return weight.clamp_min_(1e-6)
    for has_neighbor, axis, side, extent in (
        (top, 0, "rise", tile.height), (bottom, 0, "fall", tile.height),
        (left, 1, "rise", tile.width), (right, 1, "fall", tile.width),
    ):
        if not has_neighbor:
            continue
        overlap_extent = min(overlap_for(tile, tiles, axis, side), extent // 2)
        if overlap_extent <= 0:
            continue
        t = torch.arange(1, overlap_extent + 1, dtype=dtype, device=device) / overlap_extent
        ramp = t if method == "linear" else 0.5 - 0.5 * torch.cos(math.pi * t)
        if side == "fall": ramp = ramp.flip(0)
        if axis == 0 and side == "rise":
            weight[:overlap_extent, :] *= ramp[:, None]
        elif axis == 0:
            weight[-overlap_extent:, :] *= ramp[:, None]
        elif side == "rise":
            weight[:, :overlap_extent] *= ramp[None, :]
        else:
            weight[:, -overlap_extent:] *= ramp[None, :]
    return weight.clamp_min_(1e-6)


def overlap_for(tile: Tile, tiles: tuple[Tile, ...], axis: int, side: str) -> int:
    start = tile.y if axis == 0 else tile.x
    size = tile.height if axis == 0 else tile.width
    stop = start + size
    overlaps = []
    for other in tiles:
        other_start = other.y if axis == 0 else other.x
        other_size = other.height if axis == 0 else other.width
        other_stop = other_start + other_size
        if side == "rise" and other_start < start < other_stop:
            overlaps.append(other_stop - start)
        if side == "fall" and other_start < stop < other_stop:
            overlaps.append(stop - other_start)
    return max(overlaps, default=0)


def stitch_tiles(canvas_height: int, canvas_width: int, tiles: tuple[Tile, ...],
                 predictions: list[torch.Tensor], *, method: str = "cosine",
                 sigma_scale: float = 0.25) -> torch.Tensor:
    """Blend [C,F,H,W] CPU predictions into one [C,F,H,W] canvas."""
    if len(tiles) != len(predictions) or not tiles:
        raise ValueError("each planned tile must have exactly one prediction")
    first = predictions[0]
    if first.ndim != 4:
        raise ValueError("tile predictions must be [C,F,H,W]")
    output = torch.zeros((first.shape[0], first.shape[1], canvas_height, canvas_width), dtype=torch.float32)
    weights = torch.zeros((1, 1, canvas_height, canvas_width), dtype=torch.float32)
    for tile, prediction in zip(tiles, predictions):
        if prediction.shape != (first.shape[0], first.shape[1], tile.height, tile.width):
            raise ValueError("tile prediction shape does not match its planned rectangle")
        weight = blend_weights(tile, canvas_height, canvas_width, tiles, method, sigma_scale)
        output[..., tile.y:tile.y + tile.height, tile.x:tile.x + tile.width] += prediction.float() * weight
        weights[..., tile.y:tile.y + tile.height, tile.x:tile.x + tile.width] += weight
    if torch.any(weights <= 0):
        raise RuntimeError("tile plan left uncovered output pixels")
    return output / weights
