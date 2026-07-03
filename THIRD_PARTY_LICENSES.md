# Third-party licenses

Ferrytale's own source code is released under the MIT License (see
[`LICENSE`](LICENSE)). That MIT grant covers **only** Ferrytale's own code. It does
**not** relicense the third-party software, models, or bundled assets described
below — each of those keeps its own upstream license, and you must comply with all
of them when you use, modify, or redistribute Ferrytale.

The Python package licenses below were read from the locally installed package
metadata (`importlib.metadata`) — the SPDX/license-expression field where present,
otherwise the Trove license classifier. Where the metadata is generic (for example,
classifier "BSD License" without a pinned variant), that is noted. Always treat the
upstream project as authoritative if you need the precise terms.

## Python dependencies

These are the major runtime packages installed into `.venv` (text-only play needs
only `google-genai` and `prompt-toolkit`; the rest are pulled in for voice mode).
Versions reflect the environment these were verified against.

| Package | Version | License (verified) | How verified |
| --- | --- | --- | --- |
| `google-genai` | 2.8.0 | Apache-2.0 | license-expression |
| `prompt-toolkit` | 3.0.52 | BSD-3-Clause | classifier: BSD License |
| `torch` | 2.4.0 | BSD-3-Clause | license field "BSD-3" + classifier: BSD License |
| `torchaudio` | 2.4.0 | BSD (variant not pinned in metadata; classifier: BSD License — see project) | classifier: BSD License |
| `numpy` | 2.4.6 | BSD-3-Clause (also bundles 0BSD, MIT, Zlib, CC0-1.0 components) | license-expression: `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` |
| `scipy` | 1.17.1 | BSD-3-Clause | classifier: BSD License + license text |
| `onnxruntime` | 1.26.0 | MIT | license field + classifier: MIT License |
| `openwakeword` | 0.6.0 | Apache-2.0 | classifier: Apache Software License |
| `silero-vad` | 6.2.1 | MIT | classifier: MIT License |
| `sounddevice` | 0.5.0 | MIT | license field + classifier: MIT License |
| `kokoro` | 0.9.4 | Apache-2.0 | license text (Apache 2.0) + classifier: Apache Software License |
| `omnivoice` | 0.1.5 | Apache-2.0 | license-expression |
| `aec-audio-processing` | 1.0.1 | BSD-3-Clause | license-expression |
| `hf-xet` | 1.5.1 | Apache-2.0 | license-expression + classifier: Apache Software License |
| `spacy` | 3.8.14 | MIT | license field + classifier: MIT License |
| `en_core_web_sm` | 3.8.0 | MIT | license field |

Notes:

- **`scipy`** binary wheels may dynamically bundle additional native libraries
  depending on the build, each under its own terms — for example OpenBLAS and
  LAPACK (BSD-3-Clause family), the GCC runtime library
  (GPL-3.0-or-later **with** the GCC Runtime Library Exception), and libquadmath
  (LGPL-2.1-or-later). These come from the upstream SciPy distribution, not from
  Ferrytale.
- **`numpy`** is primarily BSD-3-Clause; its license expression lists additional
  permissive licenses for vendored components.
- **`torchaudio`** metadata only records the generic "BSD License" classifier and
  does not pin an SPDX variant; consult the PyTorch project for the exact text.
- **`en_core_web_sm`** is the pre-trained spaCy English model (by Explosion),
  installed as a pip wheel from the spaCy models release and used for proper-noun
  NER when building the whisper.cpp recognition prompt.

This table lists the headline runtime packages, not the full transitive dependency
tree. Each of these in turn pulls in its own dependencies, which carry their own
(predominantly permissive) licenses; inspect `.venv` or `requirements*.txt` for the
complete resolved set.

## Non-PyPI components

### whisper.cpp

- **License:** MIT
- **Source:** [`ggml-org/whisper.cpp`](https://github.com/ggml-org/whisper.cpp),
  pinned to tag **`v1.9.1`**
- **How it is used:** not vendored into this repository. The bootstrap
  (`scripts/install --voice`) clones it at install time into
  `.cache/whisper.cpp/` and compiles `whisper-cli` natively for speech-to-text in
  voice mode.

### Wake-word model (`models/wake-word/okay.onnx`, `okay.json`)

- **License / provenance:** **trained by the Ferrytale author** for the `Okay` wake
  word using [openWakeWord](https://github.com/dscripka/openWakeWord) (Apache-2.0)
  as the training framework and runtime. The resulting model is the project's own
  work, distributed as part of this repository under the project's license;
  openWakeWord's Apache-2.0 terms apply to the framework used to train and run it,
  not as a claim over the trained model.

## Bundled audio assets

The following files are bundled with the project and are **not** covered by the
package licenses above:

- `assets/confirm-cue.wav` — short input/startup cue sound
- `assets/turn-cue.wav` — short end-of-narration cue sound
- `assets/voice-clone-reference.wav` — default narration voice-clone reference clip

These are project-provided assets. Their provenance, authorship, and usage
authorization are to be documented by the maintainer. This file makes no claim
about who authored or licenses them; that information should be added here before
relying on them in any redistribution.

## External services (not redistributed)

Ferrytale calls hosted APIs at runtime that are governed by their providers' own
terms, not by any license in this repository:

- **Google Gemini** — via `google-genai`, using your `GEMINI_API_KEY`.
- **ElevenLabs Voice Design** — optional, for first-time character-voice seeding,
  using your `ELEVENLABS_API_KEY`.

Transcripts downloaded from ClubFloyd are community playthrough logs fetched on
demand and treated as local cache; they are not committed to or redistributed by
this repository.
