# Product Proposal: ReflexTTS

### 1. Rationale / Concept Justification

The traditional TTS pipeline is an open-loop system. Implementing a Multi-Agent System (MAS) allows us to apply the RLAIF (Reinforcement Learning from AI Feedback) paradigm directly during inference. Delegating quality control to a council of agents (ASR + LLM) guarantees 100% semantic alignment between the audio and the text, reducing manual QA costs.

### 2. Goals and Metrics

**Project Goal:** To create an autonomous speech synthesis pipeline capable of detecting and correcting its own acoustic and semantic errors.

* **Product Metrics:**
  * *Manual QA Time:* Reduce manual editing time for 1 hour of generated audio from 3 hours to 15 minutes.
  * *Human Acceptance Rate:* The percentage of generations accepted by the user without manual tweaks (Target: > 95%).
* **Agentic Metrics:**
  * *Average Convergence Loops:* The average number of iterations (negotiation loops) required to fix a phrase (Target: < 2.5).
  * *Critic Precision:* The accuracy of the Critic Agent in detecting real hallucinations (compared to ground truth).
* **Technical Metrics:**
  * *WER (Word Error Rate):* < 1% on the final audio output.
  * *SECS (Speaker Embedding Cosine Similarity):* Voice timbre retention across inpainting seams (Target: > 0.85).

### 3. Potential Use Cases and Edge Cases

* **Primary Scenario:** Voicing a book chapter featuring dialogue and authorial remarks.
* **Edge-case 1 (Lexical Anomalies):** A made-up fantasy name (e.g., "Glorfindel") or a complex acronym. The Actor Agent fails to pronounce it, and the Critic constantly flags it as an error. *Risk:* Infinite loop.
* **Edge-case 2 (Homographs):** Words spelled identically but pronounced differently depending on context (e.g., "read" present vs "read" past). A deterministic ASR might misrecognize correct audio. The Critic will need an LLM for semantic audio-context analysis.
* **Edge-case 3 (Acoustic Seams):** Localized rewriting (Inpainting) of a short word within a phrase may cause phase jumps or voice energy shifts at the token boundaries.

### 4. Constraints

* **Technical (SLO, p95 latency):** Since a single user request triggers a chain of inferences from heavy models (LLM -> TTS -> ASR -> LLM -> TTS-Inpainting), the `p95 latency` for a 10-second audio clip might range from 15 to 40 seconds (RTF > 3.0).
* **Operational (Budget & Compute):** High inference cost. Each reflection loop burns LLM API tokens (GPT-4o/Claude) and utilizes heavy GPU resources (requires at least 24GB VRAM to hold the TTS and Whisper-large models in memory simultaneously).

### 5. Architectural Sketch and Data Flow

**Modules:**
* `Orchestrator:` Manages the graph state.
* `Director Agent:` LLM for deep text analysis.
* `Actor Agent:` Flow-Matching TTS model (e.g., F5-TTS) for generating audio.
* `Critic Agent:` An ensemble of Whisper-v3 (transcription) and LLM (judge).
* `Editor Agent:` Modifies latent tensors and applies the inpainting mask.

**Data Flow (Delegated vs. Non-Delegated tasks):**

* **Delegated to LLM/Agents (Stochastic):** Deep context analysis, emotional tagging (SSML/Prompt), semantic error evaluation (the Critic decides if a mispronunciation is critical or an acceptable synonym), and "Regenerate / Approve" decision-making.
* **NOT Delegated (Deterministic):** Acoustic token extraction (Encodec/DAC), Forced Alignment (finding exact start/end milliseconds of a word to apply the binary inpainting mask), mathematical WER calculation (Levenshtein distance).