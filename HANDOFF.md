# Handoff — Solaris "mimoid" rebuild (2026-09-23)

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

## Next steps
1. Read `overnight.log`; check build fail reasons; `preview` the full library.
2. Eyeball `previews/mimoid_train/val_recon_*.png` vs `val_truth.png` and `interp_*.png`.
3. Write stage 2 (caption embeddings via MiniLM, same pooling as the app; diffusion prior).
4. ONNX export: replace `F.interpolate`/7-D permutes with ≤6-D per-axis reshapes
   for ORT WebGPU; verify vs torch. ORT build has `Conv3DNaive` on WebGPU.
5. Browser: replace bridge.ts/imagination path in worker/model/renderer (renderer's
   R16F 3D texture volume path can be reused).
