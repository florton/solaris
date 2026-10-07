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

## Scene walk (2026-09-27)
A scene thought gave one object on every layer: "tea in grandmas kitchen during a rainstorm"'s
whole-sentence top 24 were all teapots (0.53), while "grandma's kitchen" (cabinet, clock, stone
house, 0.44) and "a rainstorm" (clouded mountains, umbrella, 0.45) never made the pool.
- `phrases()` splits the thought at connecting words (`WALK.connect`: in, during, on, through,
  and, …; "of" doesn't split). Each phrase is encoded on its own (one extra encoder run per phrase,
  in `model.ts` `sendDream`) and gets its own neighbourhood. Phrases whose best match is under
  `minSim` 0.35 are dropped; with ≤ 1 left, the whole thought is dreamt as before.
- `allot()`: the first phrase (the subject) takes the deep layers, later phrases the shallower
  ones: [7], [5, 2], [3, 2, 2], [2, 2, 2, 1]. Tea now walks teapots -> cabinet -> foggy sky; knight
  -> dragon -> mountains; fox -> snowy mountains.
- Neighbourhoods count same-caption repeats (caption cos > `dup` 0.99) once. Identical captions
  are *not* identical shapes (the 24 "a pink teapot." forms have latent cos 0.1–0.92); only ~420
  kept forms are shape duplicates (latent cos > 0.99), so the library itself is untouched.
- `mimoid_dream.py --flat` renders the old walk: `previews/mimoid_dream/walk_flat.png` vs
  `walk_scene.png`. `dream_model.json` got the new `dream` keys (only that key changed).
- Weak spot: atmosphere words ("a storm", "a dream", "fog") pass minSim on loose matches (red
  spiral, teddy bear). That's a library gap, not a picker one.

## Library growth: settings + atmosphere (2026-09-27)
Goal: give a scene thought's middle/shallow layers somewhere to go (weather, rooms, landscapes,
ruins, light, water), in the mimoid feel, and drop what breaks it (toys, plushies, cartoons).
- **Gap scan** (`scripts/mimoid_gaps.py`, 59 phrases, best kept caption cos): before growth,
  weather 0.483 mean best (storm → mountains with clouds / a roaring lion, "lightning" → Lightning
  McQueen), light 0.585 (sun 0.48, stars 0.44, torch 0.50), water 0.590 (wave 0.46, whirlpool 0.46),
  ruins 0.629 (aqueduct 0.50), rooms 0.640 (library 0.44), landscapes 0.679 (canyon 0.53, coastline 0.52).
- **Objaverse has almost no weather**: 346 licensed candidates, mostly clouds, snowflakes and
  rainbows (point clouds, "rainbow-colored" things and water bottles are excluded per family).
- `mimoid_data.py select --extend` appends `EXTRA_FAMILIES` to `library.jsonl` (the original rows are
  byte-identical; a copy is kept as `library_base.jsonl`). Within a family the regex's matched keyword
  is round-robined (5.8k staircases / 2k catalog desk lamps can't fill it; plain "lamp" is left out).
  +8,699 objects, 36 GB to stream: weather 346, water 906, light 2000, ruins 947, rooms 2500, landscapes 2000.
- **Mood filter** `mimoid_data.MOOD` (toy, plush, teddy, cartoon, lego, minecraft, pixelated, Pokémon,
  Sonic, Mario, …; hot-air balloons pass): skipped at select, and a `mood` column in
  `mimoid_quality.py` drops existing forms at export: 1,177 otherwise-kept forms, 14,898 → 13,721.
- Pipeline for new grids (no retraining):
  - `mimoid_encode.py` appends latents to `mimoid_grid.pt` with the encoder from `mimoid_grid_ckpt.pt`
    (its decoder is bit-identical to the shipped one; it checks that re-encoding old forms reproduces
    their latents; the original is kept as `mimoid_grid_base.pt`).
  - `mimoid_quality.py` scores only rows it hasn't seen.
  - `load_all` streams shards into the memmap: concatenating in RAM needed 2× the library, and 16 GB
    doesn't fit that at 25k.
- `mimoid_grow.py` runs the chain: build → encode → captions → quality (+sheet) → export → preview
  sheets for the new families (`previews/mimoid/library_<family>.png`) → gaps → `walk_grown.png`.
  Log: `data/mimoid/grow.log`.

### First grow run (2026-09-27, 03:06–05:35): build crashed at 8,375 / 8,659
A worker hit `MemoryError` in the EDT (10 workers × 253³ grids), and the exception killed the
build, so the chain stopped before encode. The saved shards are intact; only the unflushed buffer
was lost and gets rebuilt. Keep rates per family: weather 69%, water 93%, light 86%, ruins 90%,
rooms 90%, landscapes 76% so far (mostly `flat`). 284 landscape objects left (20 built in a smoke test).
`build` now survives this: an out-of-memory object is skipped and retried on the next run (not
logged), any other worker exception is logged as that object's failure, and the buffered grids are
flushed even on a crash. Rerun `mimoid_grow.py` to finish (build resumes).
Cleanup: `library_all.npy` (4.55 GB) is stale (shard count changed); delete it before the rerun, or
the rebuild (~7 GB) runs with both copies on disk.

### Grow rerun results (2026-09-27, 19:11–19:32, all stages exit 0)
- Build finished the last 284 (185 ok, 99 skipped, mostly flat landscapes). Encode: 7,365 new latents;
  re-encoding stored forms matches to 0.0018. Library 25,992 grids.
- Gap scan, mean best caption cos before → after: weather 0.483 → 0.621, light 0.585 → 0.795,
  water 0.590 → 0.724, ruins 0.629 → 0.692, rooms 0.640 → 0.770, landscapes 0.679 → 0.738.
  Still weak: fog 0.50, mist over the sea 0.46, whirlpool 0.46, cathedral light 0.56, stars 0.55,
  a well 0.55, a canyon 0.57.
- Walks (`walk_grown.png` vs `walk_scene.png`): scene thoughts now reach atmosphere on the shallow
  layers (tea → rain cloud → kitchen → teapots; lighthouse in a storm → cloud/clouded mountain;
  server room → rooms). Single-subject thoughts are unchanged.
- Other families are on target by caption sample (lamps/candles/moons/planets, temples/arches/
  amphorae, furniture, dioramas). **Weather was noisy**: `rainbow`/`rain`/`mist` pulled in ~20 My
  Little Pony "Rainbow Dash" models, rain boots, soda/beer cans and rainbow cakes. Note: the
  `preview --families` sheets draw from *all* built grids, not the kept ones.
- `MOOD` extended (`mimoid_data.py`): my little pony, rainbow dash, no-face, Five Nights at Freddy's,
  Mickey Mouse, transformers, ice cream, cake, candy, lollipop, sushi, skateboard, kite, soda, beer,
  "can of", rain boots/jacket. Broad words (can, jacket, earring, logo) were left out: they hit good
  forms (trash-can pillars, busts in jackets). Kept 18,748 → **18,532** (animals −78, mostly FNaF
  and Mickey Mouse; weather 203 → 170). Re-exported; gap scores unchanged (the junk never won).
  Scratchpad `kept_dump.py` (kept captions per family), `junk_try.py`, `mood_new.py`.
- In the in-app browser the dream worker came up on **wasm ×4 this time: 7 forms in 2.0–2.7 s**.
- **Load fix:** the encoder's 15 s watchdog (meant for a backend hanging in `InferenceSession.create`)
  also timed the 23 MB download, so a cold/slow first load showed "model failed to load: wasm timed
  out" (it happened on a cold dev server). `worker.ts` now posts `weights` after the fetch and
  `model.ts` arms the watchdog only then. Tested with a 20 s artificial download delay: it loads.
- Permalinks are base64 (`#t=YSByYWluYm93` = "a rainbow"); a plain-text `#t=` is ignored.

## No retraining needed (checked 2026-09-27)
The browser ships only MiniLM (pretrained) and the frozen AE decoder. The AE reconstructs the new
families as well as the ones it trained on (median recon IoU: weather 0.92, light 0.88, rooms 0.87,
water 0.86, landscapes 0.83, ruins 0.82; old families 0.78–0.97). Scratchpad `iou_fam.py`.

## Batch 2: tools, insects, instruments (prepared 2026-09-27, user runs it)
- User: no thin tools (screws, screwdrivers…), only forms that come through. `EXTRA_FAMILIES` gained
  `tools` (chunky only: anvils, vises, barrels, crates, chests, hammers, axes, pickaxes, gears, padlocks,
  cauldrons, catapults, cannons, microscopes, typewriters, …; the exclude regex drops any caption
  mentioning screws, screwdrivers, nails, bolts, wrenches, pliers, scissors, saws, chisels, drills,
  keys, knives, guns), `insects` (Spider-Man, biplanes, jewellery excluded) and `instruments`.
  2,900 objects, 10.6 GB (tools 1500, insects 800, instruments 600). `MOOD` += "pixel art".
- `select --extend` now appends only the families the manifest lacks (it used to refuse once any
  extension family was present); `--dry-run` writes the picks to `library_dryrun.jsonl` instead.
- **Thickness gate considered and rejected.** `core(r)` = share of the solid surviving a morphological
  opening by r voxels (scratchpad `thick_calib.py`, `thick.npz`). Objects are scaled to fill the
  cube, so a lone screw is an ~11-voxel rod: screws core 0.98 / recon IoU 0.96, screwdrivers
  0.88 / 0.93. What loses detail is thin *parts* next to big ones (lamp posts, swords, chair legs,
  butterflies, rotors). But 31% of the library has core(1.5) < 0.3, including open-shell busts and masks
  that render fine; within that band recon IoU is p50 0.72 and 16% fall under 0.6, which the existing
  IoU ≥ 0.6 gate already removes. So keyword exclusion + the recon-IoU gate is the "comes through" check.
- Run (disk had 25 GB free; the stale 6.8 GB `library_all.npy` is deleted first, since the rebuild writes a
  .tmp next to it):
  ```
  cd C:\Users\17207\something\solaris; $env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -u scripts\mimoid_data.py select --extend; if ($LASTEXITCODE -eq 0) { Remove-Item data\mimoid\library_all.npy; .\.venv\Scripts\python.exe -u scripts\mimoid_grow.py }
  ```
  Then check `grow.log`: kept per family for tools/insects/instruments, `previews/mimoid/library_{tools,insects,instruments}.png`
  (these show all built grids, not only the kept ones).

## v0.04 viewer fixes (2026-10-07, after v0.03 feedback)
User: the tour switched too fast and needed more presets; the full form never showed; the layer dots
should be clickable; forms should all start upright and facing the same way (free orbit stays);
too many near-duplicates (cars).
- **Whole forms**: the idle drift was a sinusoid along w with a ±0.4 rad tilt, so the slice never
  rested on a layer. Now the idle drift walks the layers one at a time (`walk` in `main.ts`). It spends
  `TOUR.layerMs` 2.8 s per layer, melting over ~1 s with the slice leaning into w (`walkTilt` 0.35),
  then resting flat on the whole form. A new thought sweeps up to the deepest layer (closest match) and
  walks 6 → 0 → 6.
- Scroll snaps to the nearest layer `snapMs` 350 ms after the wheel stops; the arrow keys step one layer.
  The gauge dots are buttons (`goToLayer`: that layer, untilted).
- Tour: `holdMs` 12 s → 34 s (one full walk), 14 → 30 presets (16 new scene thoughts reaching the grown
  families; scratchpad `preset_more.py` walked 24 candidates, dropped moon/astronaut/tortoise/iceberg/
  wave/sunken ship/stone head).
- **Duplicates** (scratchpad `dup_orient_scan.py` → `dup_clusters.png`): Objaverse has the same model
  uploaded many times (one sports car ×8+, a tank, a van, a truck, 115 plain spheres). At latent cos > 0.95,
  4.0% of kept forms were redundant: light 13%, vessels 12%, ruins 6%, vehicles 2.5%. `mimoid_quality.py`
  now has a `dup` column (recomputed on every run, after the other limits): clusters at cos > `DUP_COS` 0.97
  keep their best-IoU form. 670 dropped; kept 18,532 → **17,838** (24 more from "pixel art" in `MOOD`).
  Re-exported. Old file: `data/mimoid_quality_predup.npz`. It only catches copies in the same orientation;
  rerun it after the orientation pass to catch rotated copies too.
- **Orientation (not done yet).** Grids keep each GLB's axes (glTF y-up, front +z). A random sample of 96
  kept forms (`orient_sample.png`) is mostly upright, but some lie on their back or side (a face mask
  facing up, a winged figure on its side), and the facing direction is arbitrary (cars point every way).
  The user wants every form to *start* in the same orientation (upright, facing the viewer); free orbit stays.
  Plan: render each kept form under the 24 axis-aligned rotations, score the renders with CLIP zero-shot
  against the caption ("upright", "front view"), and only change a form when the margin over identity is
  clear. Rotate the true grid (90° turns are exact index permutations on the centered 64³ grid), re-encode
  it with the `mimoid_grid_ckpt.pt` encoder (as `mimoid_encode.py` does), then quality → export. No
  retraining. Needs CLIP weights (not on this machine yet).

## Batch 2 redone: a much wider object pool (2026-10-07, user runs it)
User: cut down on tools, add a ton more objects and many more animals. It must *greatly* expand the
pool; from testing, a coke/soda bottle, a teacup and an apple didn't come through.
- Why: the library had no cups, mugs or apples at all and 1 teacup (the families never searched for
  them), and `MOOD` dropped every "soda"/"beer"/"can of" caption. Penguins/zebras/kangaroos: 16/8/4 kept.
- `MOOD` lost soda/beer/can of. `FAMILY_JUNK["weather"]` keeps them out of weather only (where rain/rainbow
  keywords pulled in cans); `mimoid_quality.py` applies it per family.
- Batch 2 (`EXTRA_FAMILIES`, the first family whose regex hits takes the object, so tableware and food
  come before fauna). Dry run: **23,800 objects, 81 GB to stream**:
  tableware 2000 (5.6k candidates), food 2500 (5.1k), fauna 6000 (13.3k), household 4000 (38k),
  structures 2000 (3.3k), machines 2500 (12.9k), plants 1500 (4.7k), people 1500 (20k),
  tools 400 (was 1500), insects 800, instruments 600. Caption samples checked; excludes added for
  UFO/shampoo (tableware), knives/swords/playgrounds (people), VW Beetle/butterfly knife (insects),
  drum-magazine guns (instruments).
- Disk: 17 GB free before the run. Each GLB is deleted after its grid is built; `library_all.npy` grows from
  6.8 to about 12 GB (the old one is deleted first). Browser: `library_lat.bin` goes from 27 MB to about 50 MB.
- Run (user, overnight; about 7–9 h):
  ```
  cd C:\Users\17207\something\solaris; $env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -u scripts\mimoid_data.py select --extend; if ($LASTEXITCODE -eq 0) { Remove-Item data\mimoid\library_all.npy; .\.venv\Scripts\python.exe -u scripts\mimoid_grow.py }
  ```

## Orientation pass (2026-10-07) — `scripts/mimoid_orient.py`
- CLIP ViT-B/16 zero-shot (weights in the HF cache) was tested first: it picked the right up axis 45% of the
  time and the facing 35%, at 2.2 s per form (scratchpad `orient_test.py`). Dropped.
- A self-supervised 3D CNN (32³ pooled grid → 24 rotations, ~1.6M params) learns the library's majority
  convention. Each form is scored under all 24 extra turns (voting). It is turned only if
  P(best) − P(as is) > `--margin`. **The prototype was weak**: 12 min (1,205 steps, CPU-bound at ~1.7
  steps/s) reached 35% up-axis accuracy on held-out turns (chance 17%), and voting was slow; the user
  stopped it. It is not in the grow chain. No orientations were written, and the hooks below are no-ops
  without `mimoid_orient.npz`.
- Background: the build already uses each GLB's own orientation (trimesh applies the scene graph), and glTF
  specifies +Y up and +Z front. Up is right for most forms (an estimated ~5–8% lie on their back or side in
  a 96-form sample: Z-up exports). Facing is arbitrary: the spec's +Z front isn't followed in practice.
- **User: only up matters, not facing.** Rewritten as an up-only net (`UpNet`, 16³ input, 1.9M params): a
  random yaw, then one of 6 tilts → predict the tilt. Scoring undoes each tilt (× 4 yaws) and reads
  P(upright). Data + rotations live on the GPU; 32³ was too slow (1.2 steps/s, ~25 min to score).
  8-min test on 25,992 forms: held-out up axis 0.46 → 0.60 and still rising; scoring takes 3 min.
  Forms that would be stood up: margin 0.5: 1,723 · 0.6: 1,134 · 0.7: 689 · 0.8: 319 · 0.9: 58.
  `previews/mimoid_orient.png`: the most confident flips are right (upside-down buildings/vases/chandeliers/
  mountains, lying rockets/bottles/lamps/busts), but around 0.5–0.75 about half are wrong (pianos and a
  fireplace laid on their backs, a palm tree and a standing person flipped).
- In tonight's chain as `mimoid_orient.py --minutes 40 --dry-run` (net + sheet + margin counts, nothing
  written). After review: `mimoid_orient.py --reuse --margin <m>` → `mimoid_encode.py` → `mimoid_quality.py`
  → `export_dream.py` (a few minutes). To undo, delete `data/mimoid_orient.npz` and rerun those three.
- Fallback if the net stays unreliable: **Orient Anything** (2024; a DINOv2 model trained on canonically oriented Objaverse renders,
  predicts azimuth/polar/roll from one image). Score a few renders per form → up + front. Needs its
  weights (ViT-L ~1.2 GB) and a run measured in hours on the 1650. Awaiting the user's call.
- Non-destructive: shards stay raw. `data/mimoid_orient.npz` maps uid → pose index (`ROTS`, raw → canonical).
  `load_all` turns the cached grids in place (`meta["rot"]` in `library_all_uids.json` records the
  current pose), `mimoid_encode.py` re-encodes turned rows (`rot` in `mimoid_grid.pt`), and
  `mimoid_quality.py` rescores them (`q["rot"]`). A rerun trains on the turned library and composes its corrections.
- `main.ts` `show()` resets yaw/pitch to the default view (0.6, 0.25) for every thought; orbiting stays free.

## Next steps
1. After batch 2: remaining library gaps are fog/mist, stars, whirlpool, wells, canyons.
2. Confirm wasm threads in real Chrome too (they came up in the in-app browser this session).
3. The melt midpoint between dissimilar poses is a fused lump by design; aligning forms (principal
   axes / center of mass) before the lerp could make melts read as bodies turning into each other.
4. Thin sheets (wings) show voxel stair-steps at 64³.
