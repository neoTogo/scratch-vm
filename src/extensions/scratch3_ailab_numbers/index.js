/**
 * AI Lab Numbers — Scratch 3 extension for the AI Lab for Kids platform.
 *
 * Copyright (c) 2026 AI Lab for Kids
 *
 * This file is an addition to a modified version of scratch-vm and is
 * distributed under the GNU Affero General Public License v3.0 only.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * Corresponding Source: https://github.com/neoTogo/scratch-vm
 */

/**
 * AI Lab Numbers — Scratch 3 Extension (self-contained, bidirectional).
 *
 * Phase 4 (Numbers Lab). Unlike the classification extensions this one runs in
 * one of two modes, chosen synchronously in the constructor from localStorage
 * (the platform stages the keys in src/scratch/bridge.ts before window.open):
 *
 *   COLLECT MODE  (`ailab-numbers-collect-mode` === 'true')
 *     Scratch is a DATA PRODUCER. Per-column `set [col] to [n]` command blocks
 *     stash values; `record this data point` posts
 *     `{ type: 'data-row', row: {...} }` on BroadcastChannel('ailab-numbers-data').
 *     NumbersWorkspace owns the platform-side listener and appends the row.
 *
 *   PREDICT MODE  (`ailab-numbers-model-ready` === 'true')
 *     Scratch is a CONSUMER. The trained TF.js regression model is
 *     reconstructed IN THIS TAB from `ailab-numbers-weights` via
 *     tf.io.fromMemory (no BroadcastChannel inference round-trip, same design
 *     as the vision/voice/text extensions). `predicted [target]` reports the
 *     denormalized model output for the current `set [col] to [n]` values.
 *
 * The block set is fixed per tab session: Scratch calls getInfo() once right
 * after the constructor, and later BLOCKSINFO_UPDATE only refreshes existing
 * blocks. So schema + mode must be read synchronously here (same constraint
 * the voice extension documents for its per-label blocks).
 */

const ArgumentType = require('../../extension-support/argument-type');
const BlockType = require('../../extension-support/block-type');

let tf;
try {
    tf = require('@tensorflow/tfjs');
} catch (e) {
    console.error('AI Lab Numbers: Could not load TF.js dependency', e);
}

const DATA_CHANNEL = 'ailab-numbers-data';

class Scratch3AILabNumbers {
    constructor (runtime) {
        this.runtime = runtime;

        this.schema = {features: [{name: 'Input'}], target: {name: 'Output'}};
        this.collectMode = false;
        this.modelReady = false;
        this.model = null;
        this.params = null;

        // Current value per column name, set by the `set [col] to [n]` blocks.
        this.values = {};

        // Producer channel for collect mode; opened lazily on first record.
        this.channel = null;

        this._readStateFromLocalStorage();

        if (!this.collectMode) {
            // Predict mode — reconstruct the model asynchronously.
            this._initModel();
        }
    }

    _readStateFromLocalStorage () {
        try {
            const storedSchema = window.localStorage.getItem('ailab-numbers-schema');
            if (storedSchema) {
                const parsed = JSON.parse(storedSchema);
                if (parsed && Array.isArray(parsed.features) && parsed.target) {
                    this.schema = parsed;
                }
            }
            this.collectMode = window.localStorage.getItem('ailab-numbers-collect-mode') === 'true';
            console.log(
                `AI Lab Numbers: mode=${this.collectMode ? 'collect' : 'predict'} ` +
                `features=[${this.schema.features.map(f => f.name).join(', ')}] ` +
                `target=${this.schema.target.name}`
            );
        } catch (e) {
            console.warn('AI Lab Numbers: LocalStorage read failed:', e);
        }
    }

    async _initModel () {
        if (!tf) return;
        try {
            const raw = window.localStorage.getItem('ailab-numbers-weights');
            if (!raw) {
                console.warn('AI Lab Numbers: no weights payload found');
                return;
            }
            const parsed = JSON.parse(raw);
            if (!parsed.topology || !parsed.weightSpecs || !parsed.weightsBase64) {
                console.warn('AI Lab Numbers: weights payload incomplete');
                return;
            }
            const weightData = this._base64ToArrayBuffer(parsed.weightsBase64);
            this.model = await tf.loadLayersModel(tf.io.fromMemory({
                modelTopology: parsed.topology,
                weightSpecs: parsed.weightSpecs,
                weightData
            }));
            this.params = parsed.params;
            this.modelReady = true;
            console.log('AI Lab Numbers: model reconstructed and ready');
            this._refreshBlocks();
        } catch (e) {
            console.error('AI Lab Numbers: model initialization failed:', e);
            this.modelReady = false;
        }
    }

    _base64ToArrayBuffer (base64) {
        const binary = window.atob(base64);
        const len = binary.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
        return bytes.buffer;
    }

    _refreshBlocks () {
        if (this.runtime && this.runtime.extensionManager) {
            this.runtime.emit('EXTENSION_DATA_LOADING', {extensionId: 'ailabNumbers'});
            this.runtime.emit('BLOCKSINFO_UPDATE', {
                id: 'ailabNumbers',
                blocks: this.getInfo().blocks,
                menus: this.getInfo().menus
            });
        }
    }

    // ─── Normalization (mirrors src/ml/regression.ts) ───
    _normalize (value, range) {
        if (!range) return 0;
        const span = range.max - range.min;
        return span === 0 ? 0 : (value - range.min) / span;
    }

    _denormalize (value, range) {
        if (!range) return value;
        return value * (range.max - range.min) + range.min;
    }

    getInfo () {
        const blocks = [];

        // ── Per-feature `set [col] to [n]` command blocks (both modes) ──
        for (let i = 0; i < this.schema.features.length; i++) {
            blocks.push({
                opcode: `setFeature_${i}`,
                blockType: BlockType.COMMAND,
                text: `set ${this.schema.features[i].name} to [VALUE]`,
                arguments: {
                    VALUE: {type: ArgumentType.NUMBER, defaultValue: 0}
                }
            });
        }

        if (this.collectMode) {
            // Collect mode also needs the target value, plus the record block.
            blocks.push({
                opcode: 'setTarget',
                blockType: BlockType.COMMAND,
                text: `set ${this.schema.target.name} to [VALUE]`,
                arguments: {
                    VALUE: {type: ArgumentType.NUMBER, defaultValue: 0}
                }
            });
            blocks.push('---');
            blocks.push({
                opcode: 'recordDataPoint',
                blockType: BlockType.COMMAND,
                text: 'record this data point'
            });
        } else {
            // Predict mode: reporter + readiness boolean.
            blocks.push('---');
            blocks.push({
                opcode: 'predictTarget',
                blockType: BlockType.REPORTER,
                text: `predicted ${this.schema.target.name}`
            });
            blocks.push({
                opcode: 'isModelReady',
                blockType: BlockType.BOOLEAN,
                text: 'is numbers model ready?'
            });
        }

        return {
            id: 'ailabNumbers',
            name: 'AI Lab Numbers',
            color1: '#3AA5B3',
            color2: '#2E8894',
            color3: '#236A74',
            blocks: blocks
        };
    }

    // ─── Block implementations ───

    setTarget (args) {
        this.values[this.schema.target.name] = Number(args.VALUE) || 0;
    }

    recordDataPoint () {
        const row = {};
        for (const f of this.schema.features) {
            row[f.name] = Number(this.values[f.name]) || 0;
        }
        row[this.schema.target.name] = Number(this.values[this.schema.target.name]) || 0;

        try {
            if (!this.channel) this.channel = new BroadcastChannel(DATA_CHANNEL);
            this.channel.postMessage({type: 'data-row', row});
        } catch (e) {
            console.warn('AI Lab Numbers: failed to post data row', e);
        }
    }

    predictTarget () {
        if (!this.modelReady || !this.model || !this.params) return 0;
        const x = this.schema.features.map(f =>
            this._normalize(Number(this.values[f.name]) || 0, this.params.features[f.name])
        );
        const out = tf.tidy(() => {
            const pred = this.model.predict(tf.tensor2d([x]));
            return pred.dataSync()[0];
        });
        const denorm = this._denormalize(out, this.params.target);
        // Kid-friendly precision; matches the platform Quick Predict display.
        return Math.round(denorm * 100) / 100;
    }

    isModelReady () {
        return this.modelReady;
    }
}

// Dynamically compile the per-feature setter opcodes on the prototype.
for (let i = 0; i < 10; i++) {
    Scratch3AILabNumbers.prototype[`setFeature_${i}`] = function (args) {
        const feature = this.schema.features[i];
        if (!feature) return;
        this.values[feature.name] = Number(args.VALUE) || 0;
    };
}

module.exports = Scratch3AILabNumbers;
