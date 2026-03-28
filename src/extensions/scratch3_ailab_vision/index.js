const ArgumentType = require('../../extension-support/argument-type');
const BlockType = require('../../extension-support/block-type');
const Video = require('../../io/video');

let tf;
let mobilenet;
try {
    tf = require('@tensorflow/tfjs');
    mobilenet = require('@tensorflow-models/mobilenet');
} catch (e) {
    console.error('AI Lab Vision: Could not load TF.js dependencies', e);
}

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
        if (!tf || !mobilenet) return;

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

            console.log('AI Lab Vision: Loading TF.js models...');
            this.baseModel = await mobilenet.load({version: 1, alpha: 0.25});
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
            const normalized = tensor.toFloat().div(127.5).sub(1);
            const batched = normalized.expandDims(0);

            const embedding = this.baseModel.infer(batched, true);
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

        // ── Recognise blocks with image input ──
        blocks.push(
            {
                opcode: 'recogniseLabel',
                blockType: BlockType.REPORTER,
                text: 'recognise image [IMAGE] (label)',
                arguments: {
                    IMAGE: {type: ArgumentType.STRING, defaultValue: 'image'}
                }
            },
            {
                opcode: 'recogniseConfidence',
                blockType: BlockType.REPORTER,
                text: 'recognise image [IMAGE] (confidence)',
                arguments: {
                    IMAGE: {type: ArgumentType.STRING, defaultValue: 'image'}
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

    recogniseLabel () {
        return this.prediction.label || '';
    }

    recogniseConfidence () {
        return Math.round((this.prediction.confidence || 0) * 100);
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
