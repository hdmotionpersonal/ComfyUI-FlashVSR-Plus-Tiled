import { app } from "../../scripts/app.js";

const SPATIAL_WIDGETS = new Set([
    "spatial_tile_height",
    "spatial_tile_width",
    "spatial_overlap",
    "spatial_blend",
    "spatial_gaussian_sigma",
]);

app.registerExtension({
    name: "FlashVSRPlusTiled.SpatialTilingToggle",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FlashVSRPlusTiledOptions") return;

        const syncOriginalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = syncOriginalConfigure?.apply(this, args);
            this._flashVsrSyncSpatialWidgets?.();
            requestAnimationFrame(() => this._flashVsrSyncSpatialWidgets?.());
            return result;
        };

        const syncOriginalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = syncOriginalCreated?.apply(this, args);
            const node = this;
            const toggle = node.widgets?.find((widget) => widget.name === "spatial_tiling");
            if (!toggle) return result;

            const sync = () => {
                const enabled = Boolean(toggle.value);
                for (const widget of node.widgets ?? []) {
                    if (!SPATIAL_WIDGETS.has(widget.name)) continue;
                    if (widget._flashVsrOriginalDisabled === undefined) {
                        widget._flashVsrOriginalDisabled = Boolean(widget.options?.disabled);
                    }
                    const disabled = !enabled || widget._flashVsrOriginalDisabled;
                    widget.disabled = disabled;
                    widget.options ??= {};
                    widget.options.disabled = disabled;
                }
                node.setDirtyCanvas?.(true, true);
                node.graph?.setDirtyCanvas?.(true, true);
            };

            node._flashVsrSyncSpatialWidgets = sync;
            const originalCallback = toggle.callback;
            toggle.callback = function (...callbackArgs) {
                const value = originalCallback?.apply(this, callbackArgs);
                sync();
                requestAnimationFrame(sync);
                return value;
            };
            sync();
            return result;
        };
    },
});
