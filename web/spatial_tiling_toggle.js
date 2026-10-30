import { app } from "../../scripts/app.js";

const SPATIAL_WIDGETS = new Set([
    "spatial_tile_height",
    "spatial_tile_width",
    "spatial_overlap",
    "spatial_blend",
    "spatial_gaussian_sigma",
]);
const VAE_WIDGETS = new Set([
    "vae_tile_height",
    "vae_tile_width",
    "vae_overlap",
]);

function nearestStep(value, step) {
    return Math.round(Number(value) / step) * step;
}

function setBoundedStep(widget, step, max, { preserveZero = false } = {}) {
    if (!widget) return;
    widget.options ??= {};
    widget.options.step = step;
    widget.options.max = max;
    const value = Number(widget.value);
    const next = preserveZero && value === 0
        ? 0
        : Math.min(max, Math.max(0, nearestStep(value, step)));
    if (next !== value) widget.value = next;
}

app.registerExtension({
    name: "FlashVSRPlusTiled.SpatialTilingToggle",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FlashVSRPlusTiledOptions") return;

        const syncOriginalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            // Older saved graphs had spatial_tile_width as the final widget.
            // Move that stored value to its new position beside tile_height.
            const info = args[0];
            const values = info?.widgets_values;
            if (Array.isArray(values) && values.length === 23 &&
                ["cosine", "linear", "gaussian"].includes(values[3])) {
                const migrated = [values[0], values[1], values[22], ...values.slice(2, 22)];
                const temporalCompression = Number(migrated[6] ?? 4);
                const oldDecodeInterval = Number(migrated[9] ?? 0);
                if (oldDecodeInterval && oldDecodeInterval % (4 * temporalCompression)) {
                    migrated[9] = oldDecodeInterval * temporalCompression;
                }
                const oldVaeValues = [migrated[19], migrated[20], migrated[21]].map(Number);
                if (!oldVaeValues.every((value) => value % 8 === 0)) {
                    migrated[19] *= 8;
                    migrated[20] *= 8;
                    migrated[21] *= 8;
                }
                info.widgets_values = migrated;
            }
            const result = syncOriginalConfigure?.apply(this, args);
            this._flashVsrSyncSpatialWidgets?.();
            this._flashVsrSyncSteps?.();
            requestAnimationFrame(() => this._flashVsrSyncSpatialWidgets?.());
            requestAnimationFrame(() => this._flashVsrSyncSteps?.());
            return result;
        };

        const syncOriginalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = syncOriginalCreated?.apply(this, args);
            const node = this;
            const widgets = node.widgets ?? [];
            const find = (name) => widgets.find((widget) => widget.name === name);
            const toggle = find("spatial_tiling");
            const vaeToggle = find("vae_tiled");

            const sync = () => {
                const spatialEnabled = Boolean(toggle?.value);
                const vaeEnabled = Boolean(vaeToggle?.value);
                for (const widget of widgets) {
                    const isSpatial = SPATIAL_WIDGETS.has(widget.name);
                    const isVae = VAE_WIDGETS.has(widget.name);
                    if (!isSpatial && !isVae) continue;
                    if (widget._flashVsrOriginalDisabled === undefined) {
                        widget._flashVsrOriginalDisabled = Boolean(widget.options?.disabled);
                    }
                    const enabled = isSpatial ? spatialEnabled : vaeEnabled;
                    const disabled = !enabled || widget._flashVsrOriginalDisabled;
                    widget.disabled = disabled;
                    widget.options ??= {};
                    widget.options.disabled = disabled;
                }
                node.setDirtyCanvas?.(true, true);
                node.graph?.setDirtyCanvas?.(true, true);
            };

            const syncSteps = () => {
                const tileHeight = Number(find("spatial_tile_height")?.value ?? 512);
                const rawWidth = Number(find("spatial_tile_width")?.value ?? tileHeight);
                const tileWidth = rawWidth || tileHeight;
                const spatialOverlap = find("spatial_overlap");
                const spatialMax = Math.max(0, Math.min(tileHeight, tileWidth) / 2 - 8);
                setBoundedStep(spatialOverlap, 8, spatialMax);

                const vaeHeight = Number(find("vae_tile_height")?.value ?? 240);
                const vaeWidth = Number(find("vae_tile_width")?.value ?? 416);
                setBoundedStep(find("vae_overlap"), 8, Math.max(0, Math.min(vaeHeight, vaeWidth) - 8));

                const compression = Number(find("temporal_compression")?.value ?? 4);
                const temporalStep = 4 * compression;
                setBoundedStep(find("decode_every_frames"), temporalStep, 16384, { preserveZero: true });
                node.setDirtyCanvas?.(true, true);
                node.graph?.setDirtyCanvas?.(true, true);
            };

            node._flashVsrSyncSpatialWidgets = sync;
            node._flashVsrSyncSteps = syncSteps;
            for (const widget of widgets) {
                if (![...SPATIAL_WIDGETS, ...VAE_WIDGETS, "spatial_tiling", "vae_tiled", "temporal_compression"].includes(widget.name)) continue;
                const originalCallback = widget.callback;
                widget.callback = function (...callbackArgs) {
                    const value = originalCallback?.apply(this, callbackArgs);
                    sync();
                    syncSteps();
                    requestAnimationFrame(sync);
                    requestAnimationFrame(syncSteps);
                    return value;
                };
            }
            sync();
            syncSteps();
            return result;
        };
    },
});
