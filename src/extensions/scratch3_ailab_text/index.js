/**
 * AI Lab Text — Scratch 3 extension for the AI Lab for Kids platform.
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

let tf;
let use;
try {
    tf = require('@tensorflow/tfjs');
    use = require('@tensorflow-models/universal-sentence-encoder');
} catch (e) {
    console.error('AI Lab Text: Could not load TF.js / USE dependencies', e);
}

class Scratch3AILabText {
    constructor (runtime) {
        this.runtime = runtime;

        this.modelReady = false;
        this.useModel = null;
        this.labels = [];
        this.centroids = {}; // Map of label -> 512-dim array
        
        // Cache predictions to avoid redundant TF.js model.embed() executions during tick evaluations
        this._cache = {};
        this.lastPrediction = null;

        // Synchronously read labels + centroids from localStorage to render static block slots
        this._readStateFromLocalStorage();

        // Asynchronously initialize model weights
        this._init();
    }

    _readStateFromLocalStorage () {
        try {
            const storedLabels = window.localStorage.getItem('ailab-language-labels');
            const storedCentroids = window.localStorage.getItem('ailab-language-centroids');
            if (storedLabels && storedCentroids) {
                const parsedLabels = JSON.parse(storedLabels);
                const parsedCentroids = JSON.parse(storedCentroids);
                
                if (Array.isArray(parsedLabels) && typeof parsedCentroids === 'object') {
                    this.labels = parsedLabels;
                    this.centroids = parsedCentroids;
                    console.log(`AI Lab Text: Synced labels: [${this.labels.join(', ')}]`);
                }
            }
        } catch (e) {
            console.warn('AI Lab Text: LocalStorage read failed:', e);
        }
    }

    async _init () {
        if (!tf || !use) return;

        try {
            // Load the Universal Sentence Encoder Lite model client-side inside the Scratch tab
            console.log('AI Lab Text: Loading Universal Sentence Encoder model…');
            this.useModel = await use.load();
            
            // Warm up the model
            const warmup = await this.useModel.embed(['warmup']);
            warmup.dispose();

            this.modelReady = true;
            console.log('AI Lab Text: Base model loaded and fully warmed up');
            
            this._refreshBlocks();
        } catch (e) {
            console.error('AI Lab Text: Model initialization failed:', e);
            this.modelReady = false;
        }
    }

    _refreshBlocks () {
        if (this.runtime && this.runtime.extensionManager) {
            this.runtime.emit('EXTENSION_DATA_LOADING', {extensionId: 'ailabText'});
            this.runtime.emit('BLOCKSINFO_UPDATE', {
                id: 'ailabText',
                blocks: this.getInfo().blocks,
                menus: this.getInfo().menus
            });
        }
    }

    getInfo () {
        const blocks = [];

        // ── HAT: when [INPUT] is <label> — one per trained class ──
        for (let i = 0; i < this.labels.length; i++) {
            blocks.push({
                opcode: `whenTextIs_${i}`,
                blockType: BlockType.HAT,
                text: `when [INPUT] is ${this.labels[i]}`,
                arguments: {
                    INPUT: {
                        type: ArgumentType.STRING,
                        defaultValue: 'hello'
                    }
                }
            });
        }
        if (this.labels.length > 0) blocks.push('---');

        // ── Reporters ──
        blocks.push(
            {
                opcode: 'classifyText',
                blockType: BlockType.REPORTER,
                text: 'classify [INPUT]',
                arguments: {
                    INPUT: {
                        type: ArgumentType.STRING,
                        defaultValue: 'hello'
                    }
                }
            },
            {
                opcode: 'confidenceOfLast',
                blockType: BlockType.REPORTER,
                text: 'confidence of last classification'
            }
        );

        // confidence of <label> for [INPUT] — one reporter per trained class.
        for (let i = 0; i < this.labels.length; i++) {
            blocks.push({
                opcode: `confidenceOf_${i}`,
                blockType: BlockType.REPORTER,
                text: `confidence of ${this.labels[i]} for [INPUT]`,
                arguments: {
                    INPUT: {
                        type: ArgumentType.STRING,
                        defaultValue: 'hello'
                    }
                }
            });
        }

        blocks.push(
            '---',
            {
                opcode: 'isModelReady',
                blockType: BlockType.BOOLEAN,
                text: 'is text model ready?'
            }
        );

        return {
            id: 'ailabText',
            name: 'AI Lab Text',
            color1: '#A294F9',
            color2: '#8174DF',
            color3: '#6154C4',
            blocks: blocks
        };
    }

    // ─── Mathematical Helpers ───
    _cosineSimilarity (a, b) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA === 0 || normB === 0) return 0;
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    _softmax (scores) {
        const maxVal = Math.max(...scores);
        const exps = scores.map(s => Math.exp(s - maxVal));
        const sum = exps.reduce((acc, v) => acc + v, 0);
        if (sum === 0) return scores.map(() => 1 / scores.length);
        return exps.map(e => e / sum);
    }

    /**
     * Internal async classifier utilizing cache to stay highly performant.
     */
    async _classify (text) {
        if (this._cache[text]) return this._cache[text];

        if (!this.modelReady || !this.useModel || this.labels.length === 0) {
            return { topClass: '', confidence: 0, confidenceByLabel: {} };
        }

        // Get embedding
        const embeddingTensor = await this.useModel.embed([text]);
        const embedding = await embeddingTensor.data();
        embeddingTensor.dispose();

        // Compute similarities
        const rawScores = [];
        for (let i = 0; i < this.labels.length; i++) {
            const label = this.labels[i];
            const centroid = this.centroids[label] || new Array(512).fill(0);
            rawScores.push(this._cosineSimilarity(embedding, centroid));
        }

        // Softmax normalization
        const confidences = this._softmax(rawScores);

        const confidenceByLabel = {};
        let topClass = '';
        let topConfidence = 0;

        for (let i = 0; i < this.labels.length; i++) {
            const label = this.labels[i];
            const conf = confidences[i];
            confidenceByLabel[label] = conf;

            if (conf > topConfidence) {
                topConfidence = conf;
                topClass = label;
            }
        }

        const result = {
            topClass,
            confidence: topConfidence,
            confidenceByLabel
        };

        this._cache[text] = result;
        this.lastPrediction = result;

        return result;
    }

    // ─── Block Implementations ───

    async classifyText (args) {
        const input = String(args.INPUT || '').trim();
        if (!input) return '';
        const res = await this._classify(input);
        return res.topClass || '';
    }

    confidenceOfLast () {
        if (!this.lastPrediction) return 0;
        return Math.round((this.lastPrediction.confidence || 0) * 100);
    }

    isModelReady () {
        return this.modelReady;
    }
}

// Dynamically compile opcode methods on the prototype
for (let i = 0; i < 50; i++) {
    Scratch3AILabText.prototype[`whenTextIs_${i}`] = async function (args) {
        const input = String(args.INPUT || '').trim();
        if (!input) return false;
        const res = await this._classify(input);
        const target = this.labels[i];
        return res.topClass === target;
    };

    Scratch3AILabText.prototype[`confidenceOf_${i}`] = async function (args) {
        const input = String(args.INPUT || '').trim();
        if (!input) return 0;
        const res = await this._classify(input);
        const target = this.labels[i];
        const conf = res.confidenceByLabel[target] || 0;
        return Math.round(conf * 100);
    };
}

module.exports = Scratch3AILabText;
