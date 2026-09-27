# Handoff — Solaris "mimoid" rebuild (2026-09-23, v0.02 retool 2026-09-26)

## Direction (decided with the user)
The v1–v3 "imagination" (procedural SDF grammar → DeepSDF MLP → random-projection
bridge) was abandoned: every thought decoded to the same egg (Gate 2/3 failed).
Rebuild, portfolio-centerpiece quality, "expansive but limited" training set:

- **Forms like the book's mimoids**: mostly inanimate (stone pillars, mountains,
  cars, buildings, rocks), plus faces/busts and some animals.
- **Library**: curated Objaverse subset (CC-BY / CC-BY-SA / CC0 only; credits kept
  in the manifest) → 64³ truncated SDF grids (int8, TRUNC 0.2, object half-extent 0.9).
- **Stage 1**: 3D conv VAE, global latent 256, decoder ~3.4M params ships to browser
  (outputs the whole 64³ grid; no full-res convs — space-to-depth/depth-to-space).
- **Stage 2 (not written yet)**: small latent diffusion prior (MLP) conditioned on
  the MiniLM pooled embedding the app already computes; trained on Cap3D captions;
  seed = hash(text) → deterministic. 4th axis idea: layer scrub = denoising
  trajectory (cloud → condensed form).
- Hardware: GTX 1650 4 GB laptop (CUDA torch 2.14+cu126 installed in .venv).
  User declined buying eGPU for now; keep models lean.

## Scripts
- `scripts/mimoid_data.py` — `index` (done), `select` (done → `data/mimoid/library.jsonl`,
  19,664 objects, ~77 GB streamed), `build` (resumable; shards `library_XXXX.npz`,
  log `library.log.jsonl`), `preview` (contact sheets → `previews/mimoid/`).
  Filters: flat (<0.12 extent ratio), scattered (largest part <60%), too thin/full.
  Pilot (`pilot.jsonl`, 237 grids) verified visually — faces/cars/pillars read well.
- `scripts/mimoid_train_ae.py` — stage 1 training; smoke-tested on pilot
  (~10 forms/s on the 1650). Outputs `data/mimoid_ae.pt` (decoder + all latents),
  probes in `previews/mimoid_train/`. `--resume` continues from `data/mimoid_ae_ckpt.pt`.

## Overnight job the user is running
`build` then `mimoid_train_ae.py --hours 10`, logged to `data/mimoid/overnight.log`.

## Overnight results (checked 2026-09-24)
- Build: 7,333 new grids, 380 skipped → library 18,597 forms (1,067 total rejects,
  mostly `flat`, then `scattered`). `library_all.npy` cache 4.9 GB.
- Training: 8 h, 20,175 steps, 17 epochs. Held-out L1 flattened at 0.0352 (0.0359 → 0.0352
  over the last 1.5 h, LR already decayed). Held-out occupancy IoU 0.72 (min 0.18).
- Reconstructions keep the overall silhouette (heads, busts, pillars, toilet) but lose
  detail: faces come back as smooth mannequin heads, thin parts get holes.
- Latent space is **not smooth enough**: posterior sigma ≈ 0.21 vs latent spread 0.79
  (beta 2e-5 was nearly a plain AE). Linear interpolation midpoints shrink to a median
  14% of the endpoint volume (they dissolve into fragments). Slerp is better (59%) but
  still thins out. Random draws from a Gaussian fitted to the latents decode mostly
  to empty space or fragments.
  Diagnostic script: `scratchpad/diag.py` (not in the repo).

## Grid-latent AE (decided 2026-09-24)
Global 256-d latent was the blur bottleneck. `mimoid_train_ae.py --arch grid`: latent
[16, 8, 8, 8], decoder 1.72M params, beta 5e-6. 1 h A/B on the same 8 h LR schedule:
held-out L1 0.0275 / IoU 0.754 (old: 0.070 at 1 h, 0.0352 / 0.72 after 8 h). Visibly
sharper (toilet lid, pillar slab, dragon snout). ~5.4 forms/s. Latent spread 0.26 vs
posterior sigma 0.86 → many channels near-unused; normalize per channel for stage 2.
Straight-line blends stay present but go hollow/lacy mid-way.
**fp16 is slower on the GTX 1650** (no tensor cores): grid AE 2.80 s/step fp16 vs
1.31 fp32 (b16); prior 4.0 vs 1.2 s/step (b128). Both scripts now default to fp32
(`--amp` opts back in). The 1 h test ran in fp16, so it was half speed.

## Stage 2 (written 2026-09-24)
- `scripts/mimoid_captions.py` → `data/mimoid_captions.npz`: Cap3D captions
  (`data/cap3d/Objaverse_files/cap3d_captions.json.gz`, ODC-By, 56 MB) for all 18,597
  forms, embedded with MiniLM pooled exactly like the app (7 layers, specials dropped,
  mean, L2). The q8 ONNX the app ships matches fp32 at cos ≈ 0.99 per layer.
  Many captions repeat ("white bust of a man.") → prior must be generative.
- `scripts/mimoid_train_prior.py`: 3D conv denoiser over the normalized 8³×16 latent
  (8.3M params, attention at 4³), FiLM condition = layer embedding + learned layer tag
  (random layer 0..6 per pair → the app can dream one form per layer), 10% null
  condition for CFG, v-prediction cosine schedule, DDIM 40 steps, EMA weights.
  Probes: `previews/mimoid_prior/prompts_*.png` (8 prompts × 3 seeds) and
  `layers_*.png` (one prompt across layers 0..6). Smoke-tested only.

## Tonight's run (user runs it)
```
cd C:\Users\17207\something\solaris; $env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -u scripts\mimoid_train_ae.py --arch grid --hours 8 --resume *> data\mimoid\grid_overnight.log; if ($LASTEXITCODE -eq 0) { .\.venv\Scripts\python.exe -u scripts\mimoid_train_prior.py --hours 6 *> data\mimoid\prior_overnight.log }
```
AE continues from the 1 h checkpoint (~7 h more, fp32) → `data/mimoid_grid.pt`;
then the prior trains up to 6 h (~18k steps) on the new latents → `data/mimoid_prior.pt`.
The prior rewrites `mimoid_prior.pt` (EMA weights) every 15 min, so Ctrl+C is safe;
previews every 20 min show whether it's still improving.

## Grid AE overnight results (checked 2026-09-25)
8 h total, 18,955 steps, fp32 at ~11.2 forms/s. Held-out L1 **0.0110**, IoU **0.869**
(global AE: 0.0352 / 0.72). Recons keep faces, ears, the toilet's buttons and the pillar's
panel; they are smoother than the truth but clearly the same object. Latent std 0.134
vs posterior sigma 0.93. Straight-line interpolation in the grid latent still dissolves
mid-way, so the prior (not the lerp) has to be what moves between forms.
The prior was at 1 h / 6 h: held-out v-loss 0.90 → 0.30 → 0.26 and latent scale 0.296.
Its samples were still shapeless blobs.

## Prior v1 results (checked 2026-09-25) — archived as `data/mimoid_prior_v1*.pt`, `previews/mimoid_prior_v1/`
6 h, 15,926 steps, held-out v-loss 0.213 (flat for the last ~2 h). Heads and cars come out
right; everything else is a blob or lace, **even for prompts that are verbatim training
captions** ("a horse", "a small house", "a stone pillar" — retrieving that caption's form
gives a clean horse/house/pillar). Guidance 1→8 barely changes samples. Samples reproduce
their own training forms no better than held-out ones (IoU 0.12 vs 0.16; random pairs 0.08)
→ underfit / weak conditioning, not memorization.
**Root cause:** the grid AE uses only 3 of 16 latent channels (0, 6, 14; raw std 0.80 / 0.69 /
0.53, the rest ≤ 0.045). Decoding with the other 13 held at their mean: IoU 0.999 vs full.
v1 diffused all 16 with one global scale → the live channels sat at std ~2.5 against unit
noise (schedule skewed to low noise, where global shape and text barely matter) and most
of the model modelled dead channels.
**Fix (v2):** `mimoid_train_prior.py` now diffuses only the live channels ([3, 8, 8, 8]),
each normalized to unit std (`--live-frac 0.25` picks them). Checkpoint stores `live`,
`latent_mean` [16], `latent_std` [16]; dead channels decode at their mean (for the ONNX
export, fold them into the decoder's first conv bias). Smoke-tested; same speed.
Also tried: retrieval + SDEdit (noise the nearest-caption form to t0, re-denoise with the
prior). Clean forms, but the v1 prior only adds holes, so not worth it yet.
Diagnostic scripts in the session scratchpad: `prior_diag.py`, `prior_fit.py`, `sdedit.py`, `dead_ch.py`.

## Tonight's run (2026-09-25, user runs it)
Prior v2, 8 h fresh (no `--resume`: v1 checkpoints have 16-channel shapes):
```
cd C:\Users\17207\something\solaris; $env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -u scripts\mimoid_train_prior.py --hours 8 *> data\mimoid\prior_v2.log
```
Judge by `previews/mimoid_prior/prompts_*.png` (same prompts/seeds/layout as v1: 6 cols =
2 prompts × 3 seeds per row) against `previews/mimoid_prior_v1/prompts_final.png`.
If still weak: the random layer (0..6) per pair gives each layer 1/7 of the training;
try layer 6 only as a test.

## Prior v2 results (checked 2026-09-26) — `data/mimoid_prior.pt`
8 h, 18,837 steps, held-out v-loss 0.946 → 0.4652 (live channels only; not comparable to
v1), still creeping down by 0.0001–0.0003 per probe at the end. Big step up over v1:
- Pillars, cars and heads come out right on 3/3 seeds; the layer walk for "a statue of a woman's
  face" gives a bust on a pedestal at every layer (v1: unrelated blobs).
- Guidance now matters: g=1 fragments, g≥3 solid; castle tower appears at g 3–8. g 3–5 is the
  sweet spot (5 slightly more solid).
- Still weak: animals (horse is a vague 4-legged shape, "a sleeping dog" fails), terrain-like
  prompts (mountain, lighthouse on a cliff) collapse to flat slabs on a base, "a small house"
  is a slab, stone face/hand are rounded blocks.
- Fit IoU (same 48+48 items as v1): train 0.156 / held-out 0.166 (v1 0.121 / 0.163; random 0.075 / 0.106).
  Layers 0/3/6 score about the same, so mixing layers isn't obviously what hurts.
Diagnostic: scratchpad `prior_v2_diag.py` → `v2_guidance.png` (12 prompts × g 1,2,3,5,8).
Backed up as `data/mimoid_prior_v2*.pt`, previews in `previews/mimoid_prior_v2/`.

Data balance is not the issue (caption keywords: animals 15.6%, buildings 17.4%, heads 14.0%,
cars 11.2%; 64 caption clusters range 42–784). Train loss ≈ held-out loss → underfit.

## Tonight's run (2026-09-26): prior v3 = v2 resumed onto a 16 h schedule
LR restarts at ~half peak (frac 0.5 of the cosine) and decays over 8 more hours. Resume tested.
```
cd C:\Users\17207\something\solaris; $env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -u scripts\mimoid_train_prior.py --hours 16 --resume *> data\mimoid\prior_v3.log
```
Compare held-out v-loss with v2's 0.4652, and rerun `prior_v2_diag.py` (fit IoU + guidance sweep).

## Prior v3 results (checked 2026-09-26) — `data/mimoid_prior.pt` = `data/mimoid_prior_v3*.pt`
16 h total, 42,328 steps. Held-out v-loss 0.4652 → **0.4580**. It improved steadily until the LR
decayed, then went flat for the last ~1 h. Fit IoU (same items/seeds): train 0.179 / held-out 0.178
at L6 (v2 0.156 / 0.166); L3 and L0 are about the same. The guidance sweep (same noise as v2) looks
almost identical to v2: the horse is a little more legged, the castle tower a little worse,
mountain/lighthouse/house are still slabs. The layer walk ("a statue of a woman's face") has more
holes at layers 1, 2 and 6 than v2's; the other layers are about the same. **More of the same training has hit diminishing returns.**
Previews + diagnostics in `previews/mimoid_prior_v3/`.

Nearest-caption retrieval (`nn_forms.png`: top-8 training forms per weak prompt):
- **Model failures** (the data has clean forms, the prior averages them): horse (140 captions,
  crisp horses), small house (gabled houses), lighthouse (tall towers), castle tower.
- **Data-faithful**: "mountain" training forms are mostly flat terrain tiles, so the slab is what
  the data says; "sleeping dog" forms are dogs lying on blankets. "stone face"/"stone hand" have
  no real matches.

Retrieval + SDEdit with v3 (`sdedit_v3.png`: top-1 retrieved form, then t0 0.3/0.45/0.6/0.75 × 2
seeds, g 3) **now works**, unlike v1: horse, house, lighthouse, head, car and castle stay clean
up to t0 ≈ 0.45 with small variations in pose and detail. At 0.6 and above, holes start to appear.
Top-1 retrieval can pick the wrong kind of form ("a stone pillar" → a wall slab), so sample
among top-k.
Scripts (session scratchpad ad056f25…): `prior_v3_diag.py`, `nn_forms.py`, `sdedit_v3.py`.

## Dreaming = retrieval + SDEdit (2026-09-26) — `scripts/mimoid_dream.py`
Chosen generation path. For each layer l of a thought: take the cosine between the pooled
embedding and every caption embedding of layer l, then pick one of the top k=8 with weight
exp((sim − best)/temp), temp 0.05. Noise that form's live latent to t0 0.4 and denoise it
with the v3 prior (16 DDIM steps, CFG g 3, conditioned on the thought + layer tag).
Deterministic and browser-portable: seed = FNV-1a(UTF-8 text); one mulberry32 stream (same as
`src/bridge.ts`, checked bit-exact against node) gives 7 pick uniforms, then 1,536 uniforms →
Box-Muller → one noise field shared by all layers. `Dreamer` class = the reference the browser
port must match.
Previews in `previews/mimoid_dream/`: `prompts.png` (L6, 5 seed variants), `thoughts.png`
(abstract thoughts × layers 0..6); `*_temp02.png` = temp 0.02, where exact-caption matches
won every seed.
- Every concrete prompt now gives a recognizable, clean form (horse, heads, cars, pillars,
  lighthouses, houses, castle towers). Seed variants give real variety wherever several
  captions score close together.
- Abstract thoughts make layer stories: "a whale drifting through fog" goes whale → iceberg →
  sardine → airplane in clouds → whale. The shallow layers (L0–L2) match on words, the deep
  ones on meaning.
- A prompt with one exact caption match ("a horse") gives the same form on all 7 layers.
- Weak spots come from the library: flat terrain tiles, multi-object scenes ("small house with
  windmill and pumpkins") and holey thin forms get retrieved as-is.
Browser cost: the live latents of all 18,597 forms are about 29 MB int8, plus the caption
embeddings (7 × 384 per form, about 50 MB int8 for all layers; layer 6 only is about 7 MB),
plus the prior (8.3M params, about 17 MB fp16).

## In the browser (2026-09-26) — `npm run dev`, http://localhost:5173
- `scripts/export_dream.py` → `public/models/dream/` (about 120 MB, untracked):
  - `decoder.onnx` (6.9 MB): the dead channels + de-normalization are a fixed 1×1×1 conv in
    front; the output is world-unit SDF, x-fastest (texImage3D order).
  - `prior.onnx` (33 MB): one guided call, CFG batch inside the graph.
  - int8 library files `library_lat.bin` (29 MB) and `library_emb.bin` (50 MB), plus
    `captions.json` and `dream_model.json`.
  - No Resize and nothing above 6-D in either graph. Both are checked against torch (≤ 2e-5).
- `src/dream.ts` (retrieval, seed stream, schedule) + `src/worker.ts` `dream()`: one layer at a
  time, each posted as it finishes, and a newer thought supersedes the one in progress. The stats line
  shows "dreamt from: <caption>" for the focal layer. The old Bridge class is gone (`bridge.ts`
  keeps `mulberry32` + `poolEmbedding`); `models/imagination/` is no longer read.
- The dreamer always runs on **wasm, 4 threads** (`crossOriginIsolated`; it was 1 thread). ORT's
  WebGPU Conv3D took ~60 s for one layer. Timing in the in-app browser: 1 thread / 16 steps 52–68 s →
  4 threads 25.6 s → 4 threads / 8 steps **9.2 s** per thought. 8 steps look the same as 16 (and 4)
  (scratchpad `steps_cmp.py`).
- The encoder on WebGPU (`?backend=webgpu`) retrieved "3mm dolly" for "a horse" (wasm: "a horse").
  Check whether the q8 encoder's WebGPU hidden states differ from wasm.
- Renderer fixes: the camera was rolled 180° (right = −x, up = −y), which never mattered for the
  cloud, but it put forms upside down; fixed in `camera()` (picking shares it). Also:
  - Body colour weights are now relative to the nearest ball (a form far from the cloud
    was near-black).
  - The aura is tightened as the form condenses (the truncated SDF painted a disc).
  - The march now has 160 steps + a minimum step.
  - The layer crossfade is sharpened: two unrelated forms blended half-way looked like
    a ghost of both.
- Python replay of the browser algorithm with the ONNX files: scratchpad `onnx_dream.py`.

## Tour + reliability retool (2026-09-26, later)
- **Silent fallback, cause 1:** ORT-web's wasm module allows one `session.run` at a time across
  *all* its sessions ("Session already started"). Encoder and dreamer shared a worker, so a
  thought typed mid-dream crashed the dream, and the piece stayed a cloud with nothing on
  screen. The dreamer now has its own worker (`src/dream-worker.ts`, FIFO queue, cancellable).
  Encoder runs are serialized in `worker.ts`, and forward requests have ids (they were all keyed 0).
- **Cause 2:** wasm `run()` computes synchronously and resolves through microtasks, so the dream
  loop never yielded, and cancel/queue messages were only seen after a whole dream (~10 s). It now
  yields to the event loop (`setTimeout 0`) before every step; a cancel lands in ~5 ms.
- **Tour** (`src/presets.ts`: 12 thoughts picked from scratchpad `preset_candidates.py`, plus
  timings):
  - Starts on load. The cloud shows ≥ 3 s, the thought condenses, holds 22 s, and never advances
    while someone interacts.
  - The next preset is encoded and dreamed while the current one is on stage.
  - A typed thought pauses the tour, cancels its unfinished dreams and permalinks itself; the tour
    resumes after 60 s idle. A shared `#t=` link plays first.
- **Visible state:** a banner above the input ("the imagination is waking… n%", "dreaming… n/7",
  "failed — cloud only"), and the HUD shows the same plus "tour n/12". Condensing and the banner
  run on a 500 ms tick, not rAF.
- **Renderer perf:**
  - One R16F texture per layer, uploaded once on arrival. The worker sends half floats, so there
    is no main-thread f32→f16 on scrub.
  - `map()` skips the 64-ball cloud loop once fully condensed.
  - The long 160-step march only runs while a form is visible (the cloud keeps 64).
  - `half.ts` also fixes a mantissa-carry bug in the old toHalf.
- Measured in the in-app browser (wasm ×4): about 10–12 s per dream, about 1.6 s per layer.
- The in-app browser pane doesn't run rAF while hidden, so the condensation looks stuck there;
  that's not an app bug.

## v0.02 retool (2026-09-26, after v0.01 feedback)
User feedback on v0.01: slow, not user-first load; a few forms repeated on unrelated prompts
("ants on a log" -> helicopter); broken forms ("a knife and an arrow"); the layer axis felt like
a slideshow, not 4D.

**Root cause of the repeats: the WebGPU encoder.** On WebGPU the q8 MiniLM's hidden states have
cosine ~0 with wasm/PyTorch on *every* layer (wasm matches PyTorch to ~3 decimals). Chrome picked
WebGPU, so retrieval (and the cloud) ran on noise. `model.ts` now always uses wasm for the encoder
(`?backend=webgpu` still forces it). In Python, retrieval had no hubness (540+/600 unique top-1s on
held-out Cap3D captions); scratchpad `retr_diag.py`.

**Dreaming = retrieval + a 4D walk (SDEdit prior retired from the browser).**
- `scripts/mimoid_quality.py` -> `data/mimoid_quality.npz`: per-form metrics on the AE reconstruction
  (largest-part share, flatness, base-plate share, recon IoU vs truth, caption commas/"and", volume;
  plus `shell`, which isn't used: open scans are thin shells yet render fine). Keeps 14,898 / 18,597.
  `--sheet` -> `previews/mimoid_quality.png` (the forms just past each limit).
- Retrieval reads only the sentence embedding (pooled layer 6). The top 24 are the thought's
  neighbourhood; layers are drawn deepest-first without replacement, weighted by
  exp((sim - best)/temp), with temp 0.02 at L6 and 0.08 at L0. Every layer is on topic and different:
  deep layers are the closest match, shallow layers looser associations. The 7-layer embedding table
  is gone (50 MB -> 5.7 MB).
- SDEdit shredded thin forms (helicopter rotors) and cost 33 MB + ~10 s. The prior isn't shipped;
  `Dreamer.sdedit` stays for diagnostics.
- **Melt morphs**: each decoded SDF is extended to a full distance field (3-D EDT beyond the ±0.2
  band, `fullSdf`). Between layers: lerp + half-way inflation (`meltAmount`: keep ≥ 1.1× the smaller
  volume, clamped to [0.06, 0.25]; in practice 0.06 almost always). Truncated-SDF lerps vanish
  half-way; the diffusion bridge (noise both to t, lerp, denoise) snaps and gets holey (scratchpad
  `morph.py`/`morph2.py`/`morph3.py`).
- Reference: `scripts/mimoid_dream.py` (`Walker`, `pick_forms`, `full_sdf`, `melt_amount`,
  `body_at`) -> `previews/mimoid_dream/walk.png` (13 slices along w per thought). Browser picks match
  on neighbourhood but not always per layer (q8 encoder + int8 index shift near-ties).
- `export_dream.py`: decoder.onnx 6.9 MB, library_emb.bin 5.7 MB, captions.json 0.9 MB,
  library_lat.bin 22.9 MB, credits.json 1.2 MB ([uid, name, author, license] per form; the readout
  credits the focal form).

**4D renderer.** All 7 full SDFs are in one R16F 3D texture (64×64×448). `formField` slices the 4D
body with a hyperplane at w = focus, tilted into x by `rotXW` (q.x = x·cos, w = focus + x·sin), and
melts between the bracketing layers (held near each layer, smoothstep 0.12–0.88). The step is
scaled by 1/√(1+16 sin²) for the steeper tilted field, with 200 march steps. Controls: scroll = w,
shift+drag / shift+scroll / `[` `]` = tilt (±0.7 rad), `0` = reset; idle drifts both.

**Load, user first.** The dreamer downloads alongside the encoder, not after it. It's ready once
decoder + index + captions (13.5 MB) are in, and the latents stream in behind (dreams Range-fetch
their 7 rows until then). A thought typed during the load is queued and goes first; the tour only
starts if nobody typed. The HUD moved top-left (5 lines ran under the input).
- **ORT wasm threads can hang**: in the in-app browser, `InferenceSession.create` with numThreads ≥ 2
  never resolved (1 thread: ready in 0.5 s). The committed v0.01 worker hung there too. `DreamHost` in
  `model.ts` tries threads, and if the worker isn't ready 4 s after its download it respawns on
  1 thread and replays held messages. One thread: decoder ~0.8 s/layer, EDT + melt ~50 ms ->
  ~6.5 s per 7-form dream (v0.01: 10–12 s on 4 threads). Threads should give ~3×.
- The in-app pane's screenshots were a zoomed crop of the top-left in the first tab. A red dot at
  the page center confirmed the render is centered; a fresh tab captured normally.

Tour presets re-picked for the walk: 14 (owl, church bell, sailing ship, dragon replace the sleeping
giant and cathedral light). Scratchpad `presets_walk.py`.

## Next steps
1. **Grow the library** (the real limit on "does the form relate to my prompt": no ants, guitars
   or anchors exist). The AE doesn't need retraining: `mimoid_grid_ckpt.pt` has the encoder. Needs:
   `select` with bigger quotas / new families (tools, instruments, insects, furniture), overnight
   `build`, an encode-only script for new grids, then captions -> quality -> export. The browser cost
   is 384 B/form for the index (latents stream / Range-fetch).
2. Check whether wasm threads come up in real Chrome (they hung in the in-app browser); if they do,
   dreams should drop to ~2–3 s.
3. The melt midpoint between dissimilar poses is a fused lump by design; aligning forms (principal
   axes / center of mass) before the lerp could make melts read as bodies turning into each other.
4. Thin sheets (wings) show voxel stair-steps at 64³.
