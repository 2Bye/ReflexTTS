# ReflexTTS: Multi-Agent System for Self-Correcting Speech Synthesis

### The Problem, Target Audience, and Current Pain Points

**The Problem:** Automating Quality Assurance (QA) and localized error correction (Audio Inpainting) during the generation of long-form audio using LLM-based TTS models.

**Target Audience:** B2B segment (audiobook publishers, game-dev dubbing studios, EdTech), indie content creators (AI podcasters).

**The Pain Point:** Modern SOTA synthesis models (Voicebox, F5-TTS, CosyVoice) sound incredibly natural but lack stability. When processing long texts, they inevitably hallucinate: skipping words, swallowing word endings, generating codec artifacts, or misinterpreting intonation. Currently, an audio editor must manually listen to 100% of the generated material, identify flaws, change the seed, regenerate chunks, and splice them together in a Digital Audio Workstation (DAW). This makes scaling AI voiceovers slow, tedious, and cost-ineffective.

### What the PoC Will Demonstrate (Demo Scope)

The Proof of Concept (PoC) will showcase a closed-loop generation pipeline:

1. The user inputs a complex paragraph of text (up to 1000 characters) and selects a voice.
2. A council of agents is triggered. The UI visualizes the log of their "negotiations":
   - **Director Agent:** Analyzes and marks up the text (emotions, pauses).
   - **Actor Agent:** Generates the initial audio draft.
   - **Critic Agent:** Transcribes the audio, compares it with the source text, and identifies errors (e.g., a skipped word).
   - **Editor Agent:** Seamlessly regenerates *only the flawed segment* (2-3 seconds) using latent inpainting.
3. The output is a flawless audio file (WER = 0) delivered "right the first time" without human intervention.

### Out of Scope (What the PoC will NOT do)

* **Real-time streaming:** The system operates in an asynchronous batch mode. Latencies < 500 ms are not supported, as agent reflection requires processing time.
* **Zero-shot Voice Cloning (User Voice):** To mitigate deepfake generation risks, the PoC will strictly use pre-installed (whitelisted), safe voices.
* **SFX Generation:** Music, singing, and background noises are out of scope.