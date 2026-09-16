/**
 * AI Lab Voice — Scratch 3 extension for the AI Lab for Kids platform.
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
 * AI Lab Voice — Scratch 3 Extension (self-contained).
 *
 * Mirrors scratch3_ailab_vision: loads its own TF.js model from IndexedDB and
 * runs inference inside the Scratch tab. No cross-tab BroadcastChannel
 * delegation — the previous design (platform tab owns the recognizer, Scratch
 * extension forwards listen-start over BroadcastChannel) broke against
 * Chrome's backgrounded-tab autoplay policy: the platform's AudioContext
 * couldn't resume from a Scratch-tab user gesture, so tr.listen() silently
 * captured nothing.
 *
 * Running in-Scratch costs ~5 MB of extra RAM (the speech-commands
 * BROWSER_FFT base model) but solves the autoplay-policy problem entirely:
 * the kid's `start listening` block click *is* the user gesture, in the same
 * tab as the AudioContext.
 *
 * The trained transfer-recognizer head is persisted by the platform at
 * `indexeddb://${slug}-voice-v1`. We load it here exactly the same way the
 * platform does, plus pull labels from the AILabDB Dexie store (the lib's
 * `wordLabels()` returns null after `load()` — a known speech-commands quirk).
 */

const BlockType = require('../../extension-support/block-type');

let tf;
let speechCommands;
try {
    tf = require('@tensorflow/tfjs');
    speechCommands = require('tensorflow-models-speech-commands');
} catch (e) {
    console.error('AI Lab Voice: Could not load TF.js / speech-commands deps', e);
}

const BG_NOISE_LABEL = '_background_noise_';
const PROBABILITY_THRESHOLD = 0.7;
const OVERLAP_FACTOR = 0.5;

class Scratch3AILabVoice {
    constructor (runtime) {
        this.runtime = runtime;

        this.modelReady = false;
        this.listening = false;
        this.baseRecognizer = null;
        this.transferRecognizer = null;
        this.labels = [];
        this.projectSlug = null;

        this.prediction = {label: '', confidence: 0};
        this.previousLabel = '';
        this.confidenceByLabel = {};

        // ── Synchronous label load from localStorage ──
        // CRITICAL: Scratch calls getInfo() immediately after the constructor
        // returns to build the toolbox. BLOCKSINFO_UPDATE later only updates
        // *existing* blocks; it doesn't add new ones. So our per-label HAT
        // and reporter blocks must exist on first getInfo() — which means
        // labels must be populated synchronously here. The platform writes
        // them to `ailab-voice-labels` in openScratchEditorForVoice before
        // window.open(). Vision uses the same trick with `ailab-labels`.
        this._readLabelsFromLocalStorage();

        // Async init for the actual model weights — only needed for inference.
        this._init();
    }

    _readLabelsFromLocalStorage () {
        try {
            const stored = window.localStorage.getItem('ailab-voice-labels');
            if (stored) {
                const parsed = JSON.parse(stored);
                if (Array.isArray(parsed) && parsed.length > 0) {
                    this.labels = parsed;
                }
            }
        } catch (e) {
            // localStorage unavailable (e.g. extension is in a true sandbox
            // worker without window access) — fall back to async IndexedDB read.
        }
    }

    async _init () {
        if (!tf || !speechCommands) return;
        try {
            // ── 1. Read the trained model's labels + project slug from AILabDB ──
            const state = await this._readLatestModelState();
            if (!state) {
                console.warn('AI Lab Voice: no trained model found in AILabDB');
                return;
            }
            this.labels = state.classLabels || [];
            this.projectSlug = state.projectSlug;

            // ── 2. Load the base BROWSER_FFT recognizer ──
            console.log('AI Lab Voice: Loading speech-commands base recognizer…');
            this.baseRecognizer = speechCommands.create('BROWSER_FFT');
            await this.baseRecognizer.ensureModelLoaded();

            // ── 3. Create a transfer recognizer for this project's slug and
            // load the trained head from IndexedDB. The slug must match what
            // the platform used when saving, so the URL resolves correctly. ──
            this.transferRecognizer = this.baseRecognizer.createTransfer(this.projectSlug);
            const modelUrl = `indexeddb://${this.projectSlug}-voice-v1`;
            await this.transferRecognizer.load(modelUrl);

            this.modelReady = true;
            console.log(
                `AI Lab Voice: Ready. Labels: [${this.labels.join(', ')}], slug: ${this.projectSlug}`
            );
            this._refreshBlocks();
        } catch (e) {
            console.error('AI Lab Voice: init failed', e);
            this.modelReady = false;
        }
    }

    async _readLatestModelState () {
        const db = await new Promise((resolve, reject) => {
            const req = window.indexedDB.open('AILabDB');
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        });

        if (!db.objectStoreNames.contains('modelState') ||
            !db.objectStoreNames.contains('projects')) {
            db.close();
            return null;
        }

        // Find the most recent VOICE project's modelState. We filter to voice
        // by joining projects.moduleType so vision projects don't get picked.
        const projectTx = db.transaction(['projects'], 'readonly');
        const projects = await new Promise((resolve) => {
            const req = projectTx.objectStore('projects').getAll();
            req.onsuccess = () => resolve(req.result || []);
            req.onerror = () => resolve([]);
        });
        const voiceProjects = projects.filter(p => p.moduleType === 'voice');
        if (voiceProjects.length === 0) { db.close(); return null; }

        const stateTx = db.transaction(['modelState'], 'readonly');
        const states = await new Promise((resolve) => {
            const req = stateTx.objectStore('modelState').getAll();
            req.onsuccess = () => resolve(req.result || []);
            req.onerror = () => resolve([]);
        });
        db.close();

        // Match modelState rows to voice projects, then pick the most recently trained.
        const voiceProjectIds = new Set(voiceProjects.map(p => p.id));
        const voiceStates = states
            .filter(s => voiceProjectIds.has(s.projectId))
            .sort((a, b) => (b.trainedAt || 0) - (a.trainedAt || 0));
        if (voiceStates.length === 0) return null;

        const state = voiceStates[0];
        const project = voiceProjects.find(p => p.id === state.projectId);
        return {
            classLabels: state.classLabels,
            projectSlug: project ? project.slug : null,
        };
    }

    _refreshBlocks () {
        if (this.runtime && this.runtime.extensionManager) {
            this.runtime.emit('EXTENSION_DATA_LOADING', {extensionId: 'ailabVoice'});
            this.runtime.emit('BLOCKSINFO_UPDATE', {
                id: 'ailabVoice',
                blocks: this.getInfo().blocks,
                menus: this.getInfo().menus
            });
        }
    }

    getInfo () {
        // Vision-style approach: generate one HAT block per trained label
        // (rather than one HAT-with-menu) because Scratch's sandboxed
        // extension worker caches static menu items at first registration —
        // BLOCKSINFO_UPDATE re-renders blocks but not menu argument lists.
        // Per-label blocks dodge this entirely: each label is its own block
        // that appears the moment getInfo() runs with labels populated.
        const userLabels = this._getUserLabels();
        const blocks = [];

        // ── HAT: when I hear <label> — one per trained class ──
        for (let i = 0; i < userLabels.length; i++) {
            blocks.push({
                opcode: `whenIHear_${i}`,
                blockType: BlockType.HAT,
                text: `when I hear ${userLabels[i]}`,
                isEdgeActivated: true
            });
        }
        if (userLabels.length > 0) blocks.push('---');

        // ── Reporters ──
        blocks.push(
            {
                opcode: 'lastHeardSound',
                blockType: BlockType.REPORTER,
                text: 'last heard sound'
            }
        );
        // confidence of <label> — one reporter per trained class.
        for (let i = 0; i < userLabels.length; i++) {
            blocks.push({
                opcode: `confidenceOf_${i}`,
                blockType: BlockType.REPORTER,
                text: `confidence of ${userLabels[i]}`
            });
        }

        blocks.push(
            '---',
            {
                opcode: 'startListening',
                blockType: BlockType.COMMAND,
                text: 'start listening'
            },
            {
                opcode: 'stopListening',
                blockType: BlockType.COMMAND,
                text: 'stop listening'
            },
            {
                opcode: 'isListening',
                blockType: BlockType.BOOLEAN,
                text: 'is listening?'
            },
            '---',
            {
                opcode: 'isModelReady',
                blockType: BlockType.BOOLEAN,
                text: 'is the voice model ready?'
            }
        );

        return {
            id: 'ailabVoice',
            name: 'AI Lab Voice',
            color1: '#E07ECC',
            color2: '#C062AC',
            color3: '#A04A8C',
            blocks: blocks
        };
    }

    _getUserLabels () {
        // Hide the background-noise sentinel from kid-facing blocks.
        return (this.labels || []).filter(l => l !== BG_NOISE_LABEL);
    }

    // ── Blocks ────────────────────────────────────────────────────────────
    // Per-label `whenIHear_<i>` and `confidenceOf_<i>` methods are defined
    // on the prototype below (Scratch dispatches by opcode name).

    lastHeardSound () {
        return this.prediction.label || '';
    }

    isListening () {
        return this.listening;
    }

    isModelReady () {
        return this.modelReady;
    }

    async startListening () {
        if (!this.modelReady || !this.transferRecognizer) {
            console.warn('AI Lab Voice: cannot start listening — model not ready');
            return;
        }
        if (this.transferRecognizer.isListening()) {
            this.listening = true;
            return;
        }
        try {
            await this.transferRecognizer.listen(
                async (result) => {
                    const scores = result.scores;
                    const flat = Array.isArray(scores) ? scores[0] : scores;
                    // wordLabels() returns null after load() — use the labels
                    // we pulled from AILabDB. Their order matches the model
                    // output indices because the platform saved them from
                    // the same wordLabels() call right after training.
                    const all = [];
                    for (let i = 0; i < this.labels.length; i++) {
                        all.push({label: this.labels[i], confidence: flat[i] || 0});
                    }
                    all.sort((a, b) => b.confidence - a.confidence);

                    // When BG noise is the highest-scored class, treat that
                    // as silence → empty top label. This lets edge-triggered
                    // `when I hear X` hats reset between utterances: speaking
                    // "laser" fires once, then silence resets previousLabel,
                    // and the next "laser" fires again. Earlier we forced a
                    // non-BG top even during silence, which made the hat
                    // fireable only once per page load.
                    const top = all[0];
                    const isSilent = top && top.label === BG_NOISE_LABEL;

                    this.previousLabel = this.prediction.label;
                    this.prediction = {
                        label: isSilent ? '' : (top ? top.label : ''),
                        confidence: isSilent ? 0 : (top ? top.confidence : 0)
                    };
                    this.confidenceByLabel = {};
                    for (const item of all) this.confidenceByLabel[item.label] = item.confidence;
                },
                {
                    probabilityThreshold: PROBABILITY_THRESHOLD,
                    overlapFactor: OVERLAP_FACTOR,
                    invokeCallbackOnNoiseAndUnknown: true
                }
            );
            this.listening = true;
            console.log('AI Lab Voice: listening started, mic should be active');
        } catch (e) {
            console.error('AI Lab Voice: startListening failed', e);
            this.listening = false;
        }
    }

    async stopListening () {
        if (!this.transferRecognizer) return;
        try {
            if (this.transferRecognizer.isListening()) {
                await this.transferRecognizer.stopListening();
            }
        } catch (e) {
            console.warn('AI Lab Voice: stopListening errored', e);
        }
        this.listening = false;
    }
}

// Dynamically create whenIHear_0..49 and confidenceOf_0..49 methods on the
// prototype. Scratch calls these by opcode name, so we register up to 50
// per-label slots and dispatch through the current user-label list.
for (let i = 0; i < 50; i++) {
    Scratch3AILabVoice.prototype[`whenIHear_${i}`] = function () {
        const userLabels = this._getUserLabels();
        const target = userLabels[i];
        if (!target) return false;
        // Edge-triggered: fire once when prediction first equals target.
        return this.prediction.label === target && this.previousLabel !== target;
    };
    Scratch3AILabVoice.prototype[`confidenceOf_${i}`] = function () {
        const userLabels = this._getUserLabels();
        const target = userLabels[i];
        if (!target) return 0;
        if (target === this.prediction.label) {
            return Math.round((this.prediction.confidence || 0) * 100);
        }
        const c = this.confidenceByLabel[target];
        return typeof c === 'number' ? Math.round(c * 100) : 0;
    };
}

module.exports = Scratch3AILabVoice;
