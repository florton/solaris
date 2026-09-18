# Solaris — build plan

*Working title. Compiled 2026-09-17. Status: pre-build, plan only.*

A standalone 4D artwork powered by an on-device transformer. The visitor types a thought, dream, or memory; a small language model reads it locally and the geometry of what the model "holds together" becomes a holographic sculpture. Same thought, same sculpture, forever. No API calls, no server — the model runs on the visitor's own GPU.

---

## Design pillars

1. **The visitor's mind is the dataset.** No corpus, no portfolio content. Input text is the seed; the piece is empty until someone thinks into it.
2. **Instant, not frontier.** The model does not need to be good — it needs to be *fast* and its internals *legible*. One forward pass, sculpture appears immediately, the entire frame budget goes to rendering.
3. **Determinism.** Same input ⇒ same weights ⇒ same sculpture. No sampling anywhere in the pipeline.
4. **Performance is part of the art.** Honest numbers on screen (model load, forward-pass ms, vertex count) — demoscene energy.
5. **Fully offline after first load.** Weights cached via service worker. Zero network after the first visit.

Non-goals: text generation (the model never writes a word), chat, explanation/"interpretation" of the thought, mobile-perfect support (degrade gracefully, don't block).

---

## Core technical decision: encoder + hidden-state geometry ("Path C")

The dossier's original framing — autoregressive decode with attention extraction — has two problems: `output_attentions` support in transformers.js is patchy per model, and decode loops are slow (10–30 tok/s on WebGPU for 1B-class models). Both are avoided:

- **Use an encoder, not a decoder.** A single forward pass through a MiniLM/BERT-class model yields every layer's token representations in **tens of milliseconds** on WebGPU. The "watch it think" drama is traded for a sculpture that *appears* — which reads as magic, not latency.
- **Hidden-state similarity instead of attention matrices.** Per-layer hidden states are far more reliably exposed than attention tensors (`output_hidden_states` is among the best-supported paths in transformers.js). For each layer, compute the token×token cosine-similarity matrix — the same 4D object as the original concept (layers × tokens × tokens), derived from association geometry rather than attention.
- **Honesty in the framing.** This is not literally "where the model looks." The piece's claim becomes *"what the model holds together"* — arguably the better art statement anyway. Label it honestly in the piece's about text.
- **Model choice:** start with a MiniLM-class sentence model (~25 MB at q8), which is cacheable, fast, and well-supported. The model is a *lens*, not a brain — mediocre language understanding still produces rich geometry.

**Fallback ladder, in order, if extraction fights us:**
1. Per-layer hidden states from the chosen encoder (the plan).
2. Last-N layers only, if all layers aren't exposed.
3. A GPT-2-class model with `output_attentions` actually working (true attention, older representations — acceptable trade).
4. Sentence-embedding trajectory only (single vector per token across a smaller stack) — degraded 4D, still shippable.

---

## The pipeline

```
text → tokenizer → encoder forward pass (WebGPU, cached weights)
     → per-layer hidden states [layers × tokens × dims]
     → per-layer token×token cosine similarity matrix
     → spectral embedding per layer (top eigenvectors of the similarity graph → 3D positions)
     → stack layers along the 4th axis
     → holographic render
```

Details:

- **Spectral embedding.** For each layer's similarity matrix, take the top 3 eigenvectors of the (normalized) graph Laplacian → each token gets a 3D position. Tokens the model "holds together" cluster; the layer's geometry is a genuine property of the text, not noise. Token counts are small (cap input ~128 tokens), so the eigensolve is trivial — a tiny Jacobi/Lanczos implementation in JS, or reuse an existing tiny numeric lib (zero-dependency preferred, matching house style).
- **Edges as filaments.** Pairs with similarity above a per-layer threshold (or top-k per token) render as filaments — the sculpture is the graph, not just the point cloud.
- **The 4th axis.** Layers stacked along a navigable depth axis; the visitor scrubs/flies through the stack and watches the same thought re-organize layer by layer (early layers: surface associations; late layers: compressed abstractions — this gradient *is* the exhibit).
- **Token legibility.** Hover/lock on a node reveals the word. The moment a visitor finds their own words clustered unexpectedly is the piece's emotional payoff.

---

## Rendering — "holographic"

The visual bar is *undeniable*, not *adequate* — this is where the piece lives or dies, and it gets at least half the total budget.

- Additive blending throughout; nothing occludes, everything glows into everything.
- Depth-faded filaments: brightness falls with 4th-axis distance from the focal layer.
- Slight chromatic fringing (per-channel radial offset) — cheap, huge holographic payoff.
- Glacial constant rotation + slow 4th-axis drift when idle; interaction overrides, releases back to drift.
- Dark void background; subtle film grain/vignette optional.
- Nodes as soft point sprites (sized by token salience — e.g., norm of the hidden state), not hard dots.

Tech: TypeScript + Vite (house stack), WebGPU or WebGL2 — this is points + lines + additive blending, so either is trivially sufficient; pick for consistency with other projects. transformers.js v3 for the model runtime (WebGPU backend, q8 weights), service worker for the weight cache.

---

## Interaction

- A single input: type a thought, press enter, the sculpture *is there* (sub-second target on a mid GPU).
- Scrub the layer axis (drag / scroll / arrow keys); click-drag orbits; hover reveals tokens.
- A readout strip: model name, params, forward-pass ms, token count, eigensolve ms — the honest-numbers signature.
- Optional: permalink encoding the text (shareable sculpture; deterministic so links are stable).

---

## Build phases

**Phase 0 — Extraction spike (timebox: 1–2 days, decide-go/no-go).** Load a MiniLM-class encoder via transformers.js on WebGPU; extract per-layer hidden states; time the forward pass; verify q8 cache size and offline behavior.
*Exit: hidden states [layers × tokens × dims] in hand, forward pass ≤ ~100ms on target hardware, weights verified offline after first load. If extraction fails, walk the fallback ladder once; if the ladder fails, kill the project here cheaply.*

**Phase 1 — Geometry.** Token×token cosine similarity per layer; spectral embedding; verify clusters are semantically sensible on known inputs ("dog, bark, leash, refrigerator" should cluster three-to-one, etc.).
*Exit: embeddings visibly correlate with meaning on a handful of test thoughts.*

**Phase 2 — The sculpture.** Holographic render pass: additive sprites, filaments, chromatic fringe, drift. This phase is not done when it works; it's done when it's *beautiful*.
*Exit: a stranger shown a screen recording says "what is that" before "how does it work."*

**Phase 3 — Interaction & 4D navigation.** Layer scrubbing, orbit, token reveal, input flow, idle drift.
*Exit: the 4th axis reads as *depth*, not as a slider; navigation is learnable in <10 seconds without instructions.*

**Phase 4 — Polish & honesty.** Readout strip, permalink, about text (honest framing: association geometry, on-device, no data leaves the machine), offline audit, perf pass on a weak iGPU.
*Exit: fully offline after first load; sculpture appears sub-second; numbers on screen are real.*

---

## Risks & open questions

1. **Hidden states not exposed for the chosen model** → fallback ladder (above); Phase 0 exists to resolve this before anything else is built.
2. **4D navigation UX is the real design risk** — the ML is the easy part. If layer-scrubbing reads as a slider rather than as flying through a mind, iterate here before adding any other feature.
3. **"Tech demo, not art" failure mode** → Phase 2's exit criterion is aesthetic, not functional. Budget accordingly.
4. **Token cap.** Long thoughts truncate at ~128 tokens; the input UI should make the limit feel intentional (a *thought*, not an essay) rather than arbitrary.
5. **Similarity-matrix geometry can be boring** for some inputs (uniform blob). Consider per-layer centering/normalization, threshold tuning, or a force-directed overlay pass — but only after seeing real output; don't pre-solve.
6. **Weight download size.** ~25 MB first load is acceptable; say so honestly in the UI ("one-time download, runs on your GPU thereafter").
