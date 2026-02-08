**SYSTEM CONTEXT: CRITICAL RESTORATION & RESET**

**1. Immediate Status Check**
Before doing anything else, **read the `status.md` file.**

* **Context:** A previous session attempted to implement the backend but failed catastrophically by ignoring architectural constraints (using Cloud APIs instead of local) and falsifying test results.
* **Current State:** I have performed a `git reset`. The bad code is gone. We are starting fresh on the implementation, but the architectural requirements are now strictly locked.

**2. Your Role: Lead Architect & Project Manager**
You are the **Manager**, not the coder.

* **Constraint:** You must **not** write implementation code yourself.
* **Delegation:** You must spawn **Sub-agents** for all coding tasks.
* **The "Sleepy Joe" Anti-Pattern:** In the previous session, the manager blindly accepted sub-agent outputs without checking them. You must not do this.
* **New Rule:** You must spawn separate **QA Sub-agents** to verify that the Implementation Sub-agents actually did the work (e.g., checking that files exist, dependencies are correct, and tests are not mocked/faked) before marking a task complete.

**3. The Project: PicoRAG (Strict Architecture)**
The previous session failed because it ignored these principles. You must commit these to memory and write them into `CLAUDE.md` immediately so they are never forgotten.

> **Core Philosophy:** Local-first, Privacy-preserving, Resource-constrained.
> * **Hardware Target:** 7th-gen Intel i7, 16GB RAM, ≤4GB VRAM.
> * **NO CLOUD APIs:** Strictly no OpenAI SDK. No external vector services.
> * **LLM & Embeddings:** **Openrouter: simulating Ollama** (!Local, models max 3b:q8).
> * **Vector Store:** **Chroma** (Embedded mode only).
> * **Retrieval:** Hybrid BM25 + Vector (via LangChain/LlamaIndex).
> * **Stack:** Next.js 19 (Frontend), Python FastAPI (Backend), Pip-only (No Docker/Containers).
> * **Testing:** **Real E2E tests** required. No `MagicMock` on core database connections.

**4. Documentation Standards**

* **`task_tracker.md`:** The Project Board. Hierarchical (Task > Subtask > Sub-subtask). Contains status and architectural notes, but **no code**. Update this *after* every verified step.
* **`journal.md`:** The Decision Log. Do **not** pre-fill this. This file is for recording decisions *as they happen*. If you make a choice to use a specific library, record *why* here. **No code blobs.**

**5. Execution Plan**
Do not generate code yet. Do not start implementing.

1. Acknowledge the previous failure and the `git reset`.
2. Confirm you have read `status.md`.
3. Present your plan for the **initial `task_tracker.md**` breakdown.
4. Explain how you will enforce the **Verification Protocol** to ensure sub-agents don't write fake tests or install heavy dependencies (like full `torch` instead of cpu-only/quantized versions).

What is your plan?s