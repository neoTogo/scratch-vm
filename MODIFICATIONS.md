# Modifications to scratch-vm

This repository is a **modified version of
[scratch-vm](https://github.com/scratchfoundation/scratch-vm)** by the Scratch
Foundation, distributed — like the original — under the **GNU Affero General
Public License v3.0 only (AGPL-3.0-only)**. See [`LICENSE`](LICENSE).

This notice is provided in accordance with AGPL-3.0 §5(a): the work carries
prominent notices stating that it has been modified, and the date of any change.

Modifications by **AI Lab for Kids** (copyright © 2026 AI Lab for Kids),
for the AI Lab for Kids K-8 machine learning platform.

## Summary of changes

| Area | Change |
|---|---|
| `src/extensions/scratch3_ailab_vision/` | **Added.** Image-classification blocks (image-source reporters, recognise label/confidence, hat blocks per label). |
| `src/extensions/scratch3_ailab_voice/` | **Added.** Sound-classification blocks. |
| `src/extensions/scratch3_ailab_text/` | **Added.** Text-classification blocks. |
| `src/extensions/scratch3_ailab_numbers/` | **Added.** Numeric data-collection and regression-prediction blocks. |
| `src/extension-support/extension-manager.js` | Registered the four `ailab*` extension IDs. |
| `package.json` | Added `tensorflow-models-speech-commands` and `@tensorflow-models/universal-sentence-encoder`. |
| `webpack.config.js` | Stubbed `fs` and `util` browser fallbacks so the speech-commands ESM bundle resolves. |

No upstream scratch-vm behaviour was altered other than the extension
registration above; the four extensions are additive.

## About the extensions

These extensions are **bridge code**, not machine learning implementations.
They exchange messages with the AI Lab for Kids platform over
`BroadcastChannel` / `localStorage` and read models the platform has already
written to IndexedDB. Model training happens in the platform, which is a
separate program communicating at arm's length and is not covered by this
license.

## Corresponding Source

The complete corresponding source for the modified Scratch used by the AI Lab
for Kids platform is available at:

- scratch-vm — <https://github.com/neoTogo/scratch-vm>
- scratch-gui — <https://github.com/neoTogo/scratch-gui>
