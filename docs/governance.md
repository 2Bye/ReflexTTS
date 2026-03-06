# Governance, Risk Management & Compliance

### 1. Risk Register

| Risk | Probability / Impact | Detection | Mitigation | Residual Risk |
| :--- | :--- | :--- | :--- | :--- |
| **Infinite Loop** (Agents fail to fix an error, burning GPU/API budget) | High / High | Monitor `retry_count` in the Orchestrator's state graph. | Hardcode `MAX_RETRIES = 3`. If exhausted, return the best audio with a flag for manual human review. | Low |
| **Prompt Injection** (User text exploits the Director's prompt to generate malicious content) | Medium / High | LLM-Guardrails on input; Regex validation. | Character escaping. Use strict JSON schemas (Pydantic/Structured Outputs) for agent communication. | Low |
| **Voice Spoofing / Deepfake** (Generating fraudulent audio) | Low (in PoC) / Critical | Compare audio reference against a Blacklist database. | Zero-shot Voice Cloning is disabled in the PoC. Only a Whitelist of 3 predefined safe voices is available. | Minimal |
| **PII Leakage** via Cloud LLM API (Critic Agent) | Medium / High | Entity monitoring via NER (Named Entity Recognition). | Local Data Masking of PII before sending text to the cloud, followed by de-anonymization before local TTS generation. | Low |

### 2. Logging and Data Privacy (PII) Policy

* **Ephemeral Data:** Raw user texts, intermediate (rejected) audio drafts, and codec states are stored purely in RAM (or tmpfs) during graph execution and are **deleted immediately** upon session completion.
* **Logging:** Only anonymized metadata (session `trace_id`, iteration count, WER, error type, consumed tokens) is sent to monitoring systems (e.g., ELK). User texts are explicitly excluded from system logs. Models are not fine-tuned on user data without explicit `opt-in`.

### 3. Injection Protection and Action Confirmation

* **Tool Calling Sandbox:** Agents do not possess Arbitrary Code Execution (RCE) privileges. Tools are strictly bound by contracts. The Critic Agent is only authorized to call one function: `submit_correction(start_time, end_time, instruction)`. Any jailbreak attempts will fail the JSON parser and trigger a safe fallback.
* **Human-in-the-Loop (HitL) for Edge Cases:** If the system hits `MAX_RETRIES` but a critical error (e.g., WER > 15%) persists, automatic audio publishing is blocked. The Orchestrator pauses the graph and requests manual confirmation from a human operator. The UI presents options: `Approve Audio with Errors` or `Edit Original Text`.