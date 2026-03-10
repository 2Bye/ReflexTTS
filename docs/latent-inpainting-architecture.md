# Architecture and Theory: Latent Audio Inpainting in ReflexTTS

This document outlines the theoretical foundation, algorithmic pipeline, and technology stack for implementing the Editor Agent. This module enables the Multi-Agent System (MAS) to autonomously and seamlessly fix TTS hallucinations without regenerating the entire phrase.

## 1. Theoretical Foundation: Why Inpainting over WAV Splicing?

Traditional audio editing (generating a new word separately and inserting it via cross-fade) inevitably leads to acoustic artifacts. The fundamental frequency (F0), signal phase, background room tone, or voice spectral energy often shift abruptly. The human ear (and quality metrics) can instantly detect these seams.

For seamless editing, we operate in the **latent space** (Mel-spectrograms or continuous embeddings).

### Why Flow-Matching instead of Auto-Regressive (AR)?
Historically, LLM-TTS models (VALL-E, AudioLM) relied on the auto-regressive prediction of discrete audio tokens (similar to text generation in GPT).
* **The AR Problem for Inpainting:** AR models generate tokens strictly from left to right ($t_1 \rightarrow t_2 \rightarrow t_3$). If we want to fix a word at $t_2$, the model considers the left context ($t_1$) but cannot "look ahead" to smoothly match the intonation of the right context ($t_3$). This results in a sharp intonational discontinuity.
* **The Flow-Matching (FM) Solution:** Modern architectures (based on Diffusion Transformers — DiT) predict a vector field that maps Gaussian noise to speech. They possess **Bidirectional Attention**. During inpainting, the model receives both left and right acoustic contexts *simultaneously*, solving a boundary value generation problem and mathematically guaranteeing a smooth transition.

---

## 2. Step-by-Step Pipeline (Agent-Driven Inpainting)

The local correction process consists of 5 deterministic steps distributed among the agents:

### Step 1: Generation & Detection (Actor + Critic)
1. **Actor Agent** generates the initial draft audio $Y_{draft}$ from text $T_{target}$.
2. **Critic Agent (ASR)** transcribes $Y_{draft}$ into text $T_{pred}$.
3. **Critic Agent (LLM)** computes the diff between $T_{target}$ and $T_{pred}$, identifying the error (e.g., the model said *"king"* instead of *"ship"*).

### Step 2: Forced Alignment / Precise Localization (Critic)
To let the Editor Agent know *exactly where* to cut the audio, forced phonetic alignment is applied.
* The audio and transcription are fed into an Aligner model.
* The output provides precise word-level timestamps in milliseconds for the flawed word *"king"*. *Example: `[2.45s : 3.10s]`*.

### Step 3: Dynamic Masking (Editor)
* Timestamps are converted into latent tensor frames (depending on the model's `hop_length`).
* A binary mask $M \in \{0, 1\}^T$ is formed, where $0$ represents the segment to be replaced and $1$ represents the frozen context.
* **Duration Mismatch Problem:** The correct word *"ship"* might have a different phonetic duration than the erroneous word *"king"*. The Editor Agent must mathematically stretch/shrink the tensor, predicting a new mask length $M_{new}$ and shifting the right acoustic context along the time axis. Otherwise, the model might try to compress a long word into a 0.65-second window, causing unnatural acceleration.

### Step 4: Flow-Matching Inpainting (Editor / Actor)
* The updated tensor (with frozen left/right contexts and noise in the middle) + the correct text *"ship"* are fed into the generator.
* An ODE solver (e.g., Euler method) denoises the mask over $N$ steps.
* At each integration step $t$, the model is strictly conditioned on the mask edges:
  $x_t = M \odot \text{Model}(x_t, T_{target}) + (1 - M) \odot x_t^{\text{orig}}$

### Step 5: Decoding & Validation
* The stitched latent tensor is passed through a Vocoder to output the final WAV signal without phase loss.
* The Orchestrator sends the audio back to the Critic Agent for re-validation. If WER = 0, the loop terminates.

---

## 3. Technology Stack (Recommended Models for 2025-2026)

For a PhD dissertation, you do not need to train foundation models from scratch. Your scientific contribution focuses on orchestration and inference modification.

| Component / Agent | Recommended Tech | Justification |
| :--- | :--- | :--- |
| **MAS Orchestrator** | `LangGraph` (LangChain) | Ideal framework for creating State Machines with reflection loops. Allows passing tensors and masks within the `State`. |
| **Director & Critic (LLM)** | `GPT-4o` / `Claude 3.5` (API) or `Llama-3-8B` (Local) | Performs semantic analysis. `Structured Outputs` (guaranteed JSON response) are strictly required for reliable error-diff passing. |
| **Critic (ASR + Aligner)** | `WhisperX` (VGG Oxford) | **Mission-critical.** Unlike base Whisper, it uses Wav2Vec2/Phoneme-based VAD for **Forced Alignment**. Outputs word-level timestamps essential for masking. |
| **Actor & Editor (TTS Engine)** | `F5-TTS` (E-Space / SWJTU) | **Current SOTA.** Flow-Matching TTS based on DiT. Natively supports Inpainting (Sway Sampling) and Zero-Shot cloning. Operates directly on Mel-spectrograms. *Alternative: `CosyVoice`.* |
| **Vocoder** | `Vocos` | Fast and lightweight vocoder. Excellent at reconstructing phase from F5-TTS Mel-spectrograms, avoiding "robotic" metallic artifacts. |

---

## 4. Areas of Scientific Novelty (Research Focus for PhD)

While programming this pipeline, you will tackle several open research problems suitable for a dissertation:

1. **Dynamic Duration Modification Algorithm:** Developing a heuristic or predictor that accurately calculates exactly how many milliseconds the mask $M$ needs to be expanded or contracted based on the phonetic composition of the old vs. new word.
2. **Boundary Bleeding Optimization:** If the mask is cut exactly at the word boundaries, micro-pauses, breaths, or room reverberation might be unnaturally truncated. Research is needed to find the optimal `masking_padding` size (e.g., expanding the mask by $\pm 50$ ms) so the diffusion process smoothly stitches the acoustics.
3. **Convergence Condition (Reward Metric):** Formulating a reward metric for the Orchestrator Agent. How do we mathematically prove that the audio is *better* after inpainting? Example composite metric: $Score = \alpha \cdot \text{WER} + \beta \cdot \text{SECS} (\text{Speaker Similarity}) + \gamma \cdot \text{FAD}$.