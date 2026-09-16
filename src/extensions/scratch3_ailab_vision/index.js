/**
 * AI Lab Vision — Scratch 3 extension for the AI Lab for Kids platform.
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

const ArgumentType = require('../../extension-support/argument-type');
const BlockType = require('../../extension-support/block-type');
const Video = require('../../io/video');

let tf;
try {
    tf = require('@tensorflow/tfjs');
} catch (e) {
    console.error('AI Lab Vision: Could not load TF.js', e);
}

// The platform (ai-lab-ml-platform/src/ml/vision.ts) uses MobileNet v2 with
// alpha=1.4 loaded directly from TFHub — its penultimate layer produces a
// 1792-dim embedding. The trained "head" model in IndexedDB expects that
// shape, so Scratch must use the same base model to do inference. The previous
// implementation used @tensorflow-models/mobilenet which defaults to v2/1.0
// (1280-dim) — that produced ValueError shape mismatches whenever a real
// recognise() block tried to run.
const MOBILENET_V2_140_URL = 'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2';
const EMBEDDING_NODE = 'module_apply_default/MobilenetV2/Logits/AvgPool';
const IMAGE_SIZE = 224;

// Sentinel prefix produced by the webcam/backdrop/custom-image reporter blocks
// and consumed by the recognise blocks below. Default IMAGE input ('image')
// is treated as the webcam source for backwards compatibility.
const SOURCE_PREFIX = '@source:';

class Scratch3AILabVision {
    constructor (runtime) {
        this.runtime = runtime;
        this.modelReady = false;
        this.baseModel = null;
        this.headModel = null;
        this.labels = [];
        this.prediction = {label: '', confidence: 0};
        this.previousLabel = '';
        this._loopInterval = null;
        this._blocksBuilt = false;

        // Read labels from URL hash synchronously (passed by AI Lab platform)
        // Format: #labels=eat,wash,love
        this._readLabelsFromHash();

        // Also load model from IndexedDB (async — fills in model + any missing labels)
        this._initModel();

        // Ensure video is on when the extension is loaded
        if (this.runtime.ioDevices && this.runtime.ioDevices.video) {
            this.runtime.ioDevices.video.enableVideo();
        }

        // Listen for model updates from AI Lab platform via BroadcastChannel
        this._setupChannel();

        // Kick off the inference loop
        this._loop();
    }

    _readLabelsFromHash () {
        try {
            // Read labels from localStorage (written by AI Lab platform before opening Scratch)
            const stored = window.localStorage.getItem('ailab-labels');
            if (stored) {
                const parsed = JSON.parse(stored);
                if (Array.isArray(parsed) && parsed.length > 0) {
                    this.labels = parsed;
                    this.modelReady = window.localStorage.getItem('ailab-model-ready') === 'true';
                    console.log(`AI Lab Vision: Labels from localStorage: [${this.labels.join(', ')}]`);
                }
            }
        } catch (e) {
            // Ignore — will fall back to IndexedDB
        }
    }

    _setupChannel () {
        try {
            this._channel = new BroadcastChannel('ailab-scratch');
            this._channel.onmessage = (event) => {
                const msg = event.data;
                if (msg.type === 'model-status') {
                    if (msg.classLabels && msg.classLabels.length > 0) {
                        const changed = JSON.stringify(this.labels) !== JSON.stringify(msg.classLabels);
                        this.labels = msg.classLabels;
                        this.modelReady = msg.ready || false;
                        if (changed) this._refreshBlocks();
                        // Reload TF model if ready
                        if (msg.ready) this._initModel();
                    }
                } else if (msg.type === 'labels-updated') {
                    if (msg.classLabels) {
                        this.labels = msg.classLabels;
                        this._refreshBlocks();
                    }
                }
            };
            // Tell the AI Lab platform we're ready
            this._channel.postMessage({type: 'scratch-ready'});
        } catch (e) {
            // BroadcastChannel not available — that's ok, we'll read from IndexedDB
        }
    }

    _refreshBlocks () {
        if (this.runtime && this.runtime.extensionManager) {
            // Request Scratch to re-call getInfo() for fresh blocks
            this.runtime.emit('EXTENSION_DATA_LOADING', {extensionId: 'ailabVision'});
            // Force block toolbox refresh
            this.runtime.emit('BLOCKSINFO_UPDATE', {
                id: 'ailabVision',
                blocks: this.getInfo().blocks,
                menus: this.getInfo().menus
            });
        }
    }

    async _initModel () {
        if (!tf) return;

        try {
            const db = await new Promise((resolve, reject) => {
                const req = window.indexedDB.open('AILabDB');
                req.onsuccess = () => resolve(req.result);
                req.onerror = () => reject(req.error);
            });

            if (!db.objectStoreNames.contains('modelState') ||
                !db.objectStoreNames.contains('projects')) {
                db.close();
                return;
            }

            const tx = db.transaction(['modelState', 'projects'], 'readonly');

            const states = await new Promise((resolve) => {
                const req = tx.objectStore('modelState').getAll();
                req.onsuccess = () => resolve(req.result || []);
                req.onerror = () => resolve([]);
            });

            db.close();

            if (states.length === 0) return;

            states.sort((a, b) => b.trainedAt - a.trainedAt);
            const state = states[0];
            const labelsChanged = JSON.stringify(this.labels) !== JSON.stringify(state.classLabels);
            this.labels = state.classLabels;

            if (labelsChanged) this._refreshBlocks();

            // Get project slug to load the TF model
            const db2 = await new Promise((resolve, reject) => {
                const req = window.indexedDB.open('AILabDB');
                req.onsuccess = () => resolve(req.result);
                req.onerror = () => reject(req.error);
            });
            const tx2 = db2.transaction(['projects'], 'readonly');
            const project = await new Promise((resolve) => {
                const req = tx2.objectStore('projects').get(state.projectId);
                req.onsuccess = () => resolve(req.result);
                req.onerror = () => resolve(null);
            });
            db2.close();

            if (!project) return;

            console.log('AI Lab Vision: Loading MobileNet v2/1.4 from TFHub…');
            const graphModel = await tf.loadGraphModel(MOBILENET_V2_140_URL, {fromTFHub: true});

            // Warm up the model so the first real classification isn't cold-path slow.
            const warmup = tf.tidy(() => graphModel.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])));
            if (warmup instanceof tf.Tensor) {
                await warmup.data();
                warmup.dispose();
            }

            // Wrap the raw graph model so it matches the .infer(img, embedding)
            // API the rest of this extension was written against. The penultimate
            // node yields the 1792-dim embedding the trained head expects.
            this.baseModel = {
                infer: (img) => tf.tidy(() => {
                    const normalized = tf.mul(tf.cast(img, 'float32'), 1 / 255);
                    let resized;
                    if (img.shape[0] !== IMAGE_SIZE || img.shape[1] !== IMAGE_SIZE) {
                        resized = img.rank === 3
                            ? tf.image.resizeBilinear(normalized, [IMAGE_SIZE, IMAGE_SIZE], true)
                            : tf.image.resizeBilinear(normalized, [IMAGE_SIZE, IMAGE_SIZE], true);
                    } else {
                        resized = normalized;
                    }
                    const batched = resized.rank === 3 ? tf.expandDims(resized, 0) : resized;
                    const internal = graphModel.execute(batched, EMBEDDING_NODE);
                    return tf.squeeze(internal, [1, 2]);
                })
            };

            this.headModel = await tf.loadLayersModel(`indexeddb://${project.slug}-v1`);
            this.modelReady = true;
            console.log(`AI Lab Vision: Ready. Labels: [${this.labels.join(', ')}]`);
        } catch (e) {
            console.error('AI Lab Vision: Failed to init model', e);
        }
    }

    _loop () {
        const loopTime = Math.max(this.runtime.currentStepTime || 33, 100);
        this._loopInterval = setTimeout(this._loop.bind(this), loopTime);

        if (!this.modelReady || !this.baseModel || !this.headModel || !this.labels.length) {
            return;
        }

        if (this.runtime.ioDevices && this.runtime.ioDevices.video) {
            const frame = this.runtime.ioDevices.video.getFrame({
                format: Video.FORMAT_IMAGE_DATA,
                dimensions: [224, 224]
            });

            if (frame) {
                this._predictFrame(frame);
            }
        }
    }

    _predictFrame (imageData) {
        tf.tidy(() => {
            const tensor = tf.browser.fromPixels(imageData);
            // Pass raw [0,255] pixels — mobilenet.infer() normalizes internally
            const embedding = this.baseModel.infer(tensor, true);
            const prediction = this.headModel.predict(embedding);
            const probabilities = prediction.dataSync();

            let maxProb = 0;
            let maxIndex = 0;
            for (let i = 0; i < probabilities.length; i++) {
                if (probabilities[i] > maxProb) {
                    maxProb = probabilities[i];
                    maxIndex = i;
                }
            }

            this.previousLabel = this.prediction.label;
            this.prediction = {
                label: this.labels[maxIndex],
                confidence: maxProb
            };
        });
    }

    getInfo () {
        const blocks = [];

        // ── Image-source reporters (new in V3 — feed into the recognise blocks) ──
        blocks.push(
            {
                opcode: 'webcamImage',
                blockType: BlockType.REPORTER,
                text: 'webcam image'
            },
            {
                opcode: 'backdropImage',
                blockType: BlockType.REPORTER,
                text: 'backdrop image'
            },
            {
                opcode: 'costumeImage',
                blockType: BlockType.REPORTER,
                text: 'costume image'
            },
            {
                opcode: 'customImage',
                blockType: BlockType.REPORTER,
                text: 'custom image [URL]',
                arguments: {
                    URL: {type: ArgumentType.STRING, defaultValue: 'https://...'}
                }
            },

            '---'
        );

        // ── Recognise blocks: accept the source reporters above as IMAGE input ──
        blocks.push(
            {
                opcode: 'recogniseLabel',
                blockType: BlockType.REPORTER,
                text: 'recognise image [IMAGE] (label)',
                arguments: {
                    IMAGE: {type: ArgumentType.STRING, defaultValue: 'webcam image'}
                }
            },
            {
                opcode: 'recogniseConfidence',
                blockType: BlockType.REPORTER,
                text: 'recognise image [IMAGE] (confidence)',
                arguments: {
                    IMAGE: {type: ArgumentType.STRING, defaultValue: 'webcam image'}
                }
            },

            '---'
        );

        // ── Per-label value blocks (one reporter per trained label) ──
        for (let i = 0; i < this.labels.length; i++) {
            blocks.push({
                opcode: `whenLabel${i}`,
                blockType: BlockType.REPORTER,
                text: this.labels[i]
            });
        }

        if (this.labels.length > 0) {
            blocks.push('---');
        }

        // ── Training blocks ──
        blocks.push(
            {
                opcode: 'addTrainingData',
                blockType: BlockType.COMMAND,
                text: 'add training data [IMAGE] [LABEL]',
                arguments: {
                    IMAGE: {type: ArgumentType.STRING, defaultValue: 'image'},
                    LABEL: {type: ArgumentType.STRING, menu: 'labelsMenu'}
                }
            },
            {
                opcode: 'trainModel',
                blockType: BlockType.COMMAND,
                text: 'train new machine learning model'
            },

            '---',

            // ── Status ──
            {
                opcode: 'isModelReady',
                blockType: BlockType.BOOLEAN,
                text: 'is the machine learning model ready?'
            }
        );

        return {
            id: 'ailabVision',
            name: 'AI Lab Vision',
            color1: '#4A6CD4',
            color2: '#3A5BC4',
            color3: '#2A4AB4',
            blocks: blocks,
            menus: {
                labelsMenu: {
                    acceptReporters: false,
                    items: this._getLabelMenuItems()
                }
            }
        };
    }

    _getLabelMenuItems () {
        if (this.labels.length === 0) {
            return [{text: '(no labels)', value: '__none__'}];
        }
        return this.labels.map(label => ({text: label, value: label}));
    }

    // ── Block implementations ──

    // ── Image-source reporters: each returns a sentinel identifier that the
    // recognise blocks below decode. Returning strings keeps things type-safe
    // in Scratch's block plumbing (which marshals all values through strings).

    webcamImage () {
        return `${SOURCE_PREFIX}webcam`;
    }

    backdropImage () {
        return `${SOURCE_PREFIX}backdrop`;
    }

    costumeImage (args, util) {
        const target = util && util.target;
        if (!target) return `${SOURCE_PREFIX}costume:__none__`;
        const targetId = target.id || '__none__';
        const costumeIdx = (typeof target.currentCostume === 'number') ? target.currentCostume : 0;
        return `${SOURCE_PREFIX}costume:${targetId}:${costumeIdx}`;
    }

    customImage (args) {
        const url = String(args.URL || '').trim();
        if (!url) return `${SOURCE_PREFIX}webcam`;
        return `${SOURCE_PREFIX}custom:${url}`;
    }

    // ── Recognise dispatcher ──
    // Async so backdrop/costume/custom inference actually completes before
    // the reporter returns. Webcam case stays effectively-sync because the
    // live _loop() keeps this.prediction fresh.

    async recogniseLabel (args) {
        const pred = await this._getPredictionForSource(args && args.IMAGE);
        return pred.label || '';
    }

    async recogniseConfidence (args) {
        const pred = await this._getPredictionForSource(args && args.IMAGE);
        return Math.round((pred.confidence || 0) * 100);
    }

    async _getPredictionForSource (rawImageArg) {
        const sourceKey = this._parseSourceKey(rawImageArg);
        if (sourceKey === `${SOURCE_PREFIX}webcam`) {
            return this.prediction;
        }
        try {
            return await this._runInferenceForSource(sourceKey);
        } catch (e) {
            console.warn('AI Lab Vision: inference failed for source', sourceKey, e);
            return {label: '', confidence: 0};
        }
    }

    async _runInferenceForSource (sourceKey) {
        if (!this.modelReady || !this.baseModel || !this.headModel) {
            return {label: '', confidence: 0};
        }
        if (sourceKey === `${SOURCE_PREFIX}backdrop`) {
            return this._predictFromStageBackdrop();
        }
        if (sourceKey.indexOf(`${SOURCE_PREFIX}costume:`) === 0) {
            const rest = sourceKey.slice((`${SOURCE_PREFIX}costume:`).length);
            const lastColon = rest.lastIndexOf(':');
            const targetId = rest.slice(0, lastColon);
            const costumeIdx = parseInt(rest.slice(lastColon + 1), 10);
            return this._predictFromCostume(targetId, costumeIdx);
        }
        if (sourceKey.indexOf(`${SOURCE_PREFIX}custom:`) === 0) {
            const url = sourceKey.slice((`${SOURCE_PREFIX}custom:`).length);
            return this._predictFromUrl(url);
        }
        return {label: '', confidence: 0};
    }

    _parseSourceKey (raw) {
        const s = String(raw == null ? '' : raw);
        // Default / legacy / typed-string inputs all map to webcam.
        if (!s || s === 'image' || s === 'webcam image' || s === `${SOURCE_PREFIX}webcam`) {
            return `${SOURCE_PREFIX}webcam`;
        }
        if (s === `${SOURCE_PREFIX}backdrop` || s === 'backdrop image') {
            return `${SOURCE_PREFIX}backdrop`;
        }
        if (s.indexOf(`${SOURCE_PREFIX}costume:`) === 0) return s;
        if (s.indexOf(`${SOURCE_PREFIX}custom:`) === 0) return s;
        // Anything else: treat as a URL ('custom image' with a typed URL).
        return `${SOURCE_PREFIX}custom:${s}`;
    }

    /**
     * Classify the stage's current backdrop costume — uses the costume's asset
     * data URI rather than the composited renderer canvas, so the result
     * matches what the kid sees when they upload the same image to the
     * platform workspace.
     */
    async _predictFromStageBackdrop () {
        const stage = this.runtime && this.runtime.getTargetForStage && this.runtime.getTargetForStage();
        if (!stage) return {label: '', confidence: 0};
        const costume = stage.getCurrentCostume ? stage.getCurrentCostume() : null;
        if (!costume) return {label: '', confidence: 0};
        return this._predictFromCostumeAsset(costume);
    }

    async _predictFromCostume (targetId, costumeIdx) {
        if (!this.runtime || !this.runtime.targets) return {label: '', confidence: 0};
        const target = this.runtime.targets.find(t => t.id === targetId);
        if (!target) return {label: '', confidence: 0};
        const costumes = target.getCostumes ? target.getCostumes() : (target.sprite && target.sprite.costumes);
        const costume = costumes && costumes[costumeIdx];
        if (!costume) return {label: '', confidence: 0};
        return this._predictFromCostumeAsset(costume);
    }

    async _predictFromCostumeAsset (costume) {
        const asset = costume.asset;
        if (!asset || typeof asset.encodeDataURI !== 'function') {
            return {label: '', confidence: 0};
        }
        return this._predictFromUrl(asset.encodeDataURI());
    }

    async _predictFromUrl (url) {
        if (!url) return {label: '', confidence: 0};
        const imageData = await new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => {
                try {
                    const off = document.createElement('canvas');
                    off.width = 224;
                    off.height = 224;
                    const ctx = off.getContext('2d');
                    ctx.drawImage(img, 0, 0, off.width, off.height);
                    resolve(ctx.getImageData(0, 0, off.width, off.height));
                } catch (e) {
                    reject(e);
                }
            };
            img.onerror = reject;
            img.src = url;
        });
        return this._classifyImageData(imageData);
    }

    _classifyImageData (imageData) {
        if (!tf || !this.baseModel || !this.headModel || !this.labels.length) {
            return {label: '', confidence: 0};
        }
        return tf.tidy(() => {
            const tensor = tf.browser.fromPixels(imageData);
            const embedding = this.baseModel.infer(tensor, true);
            const prediction = this.headModel.predict(embedding);
            const probabilities = prediction.dataSync();
            let maxProb = 0;
            let maxIndex = 0;
            for (let i = 0; i < probabilities.length; i++) {
                if (probabilities[i] > maxProb) {
                    maxProb = probabilities[i];
                    maxIndex = i;
                }
            }
            return {label: this.labels[maxIndex], confidence: maxProb};
        });
    }

    isModelReady () {
        return this.modelReady;
    }

    addTrainingData (args) {
        if (this._channel) {
            this._channel.postMessage({
                type: 'add-training-data',
                image: String(args.IMAGE),
                label: String(args.LABEL)
            });
        }
    }

    trainModel () {
        if (this._channel) {
            this._channel.postMessage({type: 'train-model'});
        }
    }
}

// Dynamically create whenLabel0..whenLabel49 methods on the prototype
// Each returns the label string as a value (reporter block)
for (let i = 0; i < 50; i++) {
    Scratch3AILabVision.prototype[`whenLabel${i}`] = function () {
        if (i < this.labels.length) {
            return this.labels[i];
        }
        return '';
    };
}

module.exports = Scratch3AILabVision;
