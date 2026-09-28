# Mimoid library: a curated slice of Objaverse turned into 64³ signed distance
# grids — the forms the ocean has "seen" and can imitate. Stages:
#   index    download the 160 Objaverse metadata shards once, keep only what we
#            need (license, glb size, face count, name, author, url)
#   select   bucket Cap3D captions (+ LVIS labels) into families, filter to
#            commercial-friendly licenses (CC-BY / CC-BY-SA / CC0) and bounded
#            file sizes, fill per-family quotas deterministically;
#            --extend appends the settings/atmosphere families instead
#   build    stream each GLB, convert to a 64³ truncated SDF (int8), delete the
#            GLB; resumable, sharded
#   preview  render a contact sheet per family (GPU sphere tracing)
# Run: .venv/Scripts/python scripts/mimoid_data.py <stage> [options]
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OBJ_DIR = ROOT / "data" / "objaverse"
OUT_DIR = ROOT / "data" / "mimoid"
PREVIEW_DIR = ROOT / "previews" / "mimoid"
HF = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"

G = 64                 # shipped grid resolution
G_FINE = 2 * G - 1     # sign grid; G is exactly every other fine sample
G_UDF = 4 * G - 3      # distance grid; G_FINE is every other sample, G every fourth
H_UDF = 2.0 / (G_UDF - 1)
EXTENT = 0.9           # object's largest half-extent inside the [-1, 1] cube
TRUNC = 0.2            # stored SDF is clipped to ±TRUNC, quantized to int8
H = 2.0 / (G - 1)
H_FINE = 2.0 / (G_FINE - 1)
OFFSET = 0.6 * H       # dilation: thin sheets get ≥ ~1 voxel of thickness
SHARD = 256

LICENSES = {"by", "by-sa", "cc0"}
MAX_MB = 15.0
MAX_FACES = 400_000

# family -> (quota, regex on the caption's subject phrase)
FAMILIES = {
    "faces":     (2500, r"\b(bust|busts|head|heads|face|faces|portrait|skull|mask|masks)\b"),
    "figures":   (2000, r"\b(statue|statues|sculpture|figurine|angel|goddess|buddha|idol|torso)\b"),
    "animals":   (2500, r"\b(dog|cat|horse|lion|tiger|elephant|whale|shark|dolphin|bird|owl|eagle|fish|bear|wolf|fox|deer|cow|pig|rabbit|frog|turtle|dinosaur|octopus|snake|crab|giraffe|monkey|gorilla|camel|sheep|goat|bull|dragon|animal|lizard|crocodile|rhinoceros|hippo|bat|squirrel|mouse|rat)\b"),
    "vehicles":  (2500, r"\b(car|cars|truck|bus|van|train|locomotive|boat|ship|airplane|jet|helicopter|tank|tractor|motorcycle|submarine|spaceship|automobile|sedan|jeep)\b"),
    "pillars":   (1500, r"\b(column|columns|pillar|pillars|obelisk|monolith|monument|totem|gravestone|tombstone|headstone|stele|menhir|megalith)\b"),
    # named landforms only: generic "terrain"/"landscape" captions are mostly flat map tiles
    "terrain":   (2500, r"\b(mountain|mountains|cliff|cliffs|volcano|canyon|island|mesa|butte|hill|hills|glacier|cave|crater|peak|peaks|iceberg|stalagmite|stalactite|rock formation)\b"),
    "rocks":     (2500, r"\b(rock|rocks|boulder|boulders|stone|stones|crystal|crystals|gem|geode|mineral|pebble|meteorite|asteroid|fossil)\b"),
    "buildings": (2500, r"\b(building|house|tower|church|cathedral|temple|castle|arch|ruins|ruin|bridge|pyramid|lighthouse|dome|mosque|shrine|palace)\b"),
    "vessels":   (1000, r"\b(vase|urn|bowl|jar|pot|bell|amphora|chalice|goblet|teapot|jug)\b"),
    "organic":   (1000, r"\b(tree|mushroom|coral|seashell|shell|flower|cactus|skeleton|bone|bones|root|driftwood)\b"),
}
# LVIS labels are cleaner than captions; they add to (never override) buckets
LVIS_TO_FAMILY = {
    "mask": "faces", "sculpture": "figures", "statue_(sculpture)": "figures",
    "gravestone": "pillars", "milestone": "pillars", "gemstone": "rocks",
    "vase": "vessels", "bowl": "vessels", "bell": "vessels", "seashell": "organic",
    "mushroom": "organic", "clock_tower": "buildings", "water_tower": "buildings",
    "race_car": "vehicles", "car_(automobile)": "vehicles", "pickup_truck": "vehicles",
    "bus_(vehicle)": "vehicles", "army_tank": "vehicles", "airplane": "vehicles",
    "boat": "vehicles", "truck": "vehicles", "school_bus": "vehicles", "bullet_train": "vehicles",
    "train_(railroad_vehicle)": "vehicles", "cargo_ship": "vehicles", "passenger_ship": "vehicles",
}
LVIS_ANIMALS = {"owl", "lion", "rabbit", "elephant", "crab_(animal)", "shark", "wolf", "frog",
                "penguin", "butterfly", "giraffe", "horse", "turtle", "cat", "dog", "bird", "fish",
                "monkey", "gorilla", "bat_(animal)", "tiger", "cow", "chicken_(animal)", "deer",
                "octopus_(animal)", "goldfish", "bear", "dolphin", "horned_cow", "hummingbird",
                "polar_bear", "bulldog", "pigeon", "seahorse", "camel", "sheep", "shepherd_dog",
                "lamb_(animal)", "pug-dog", "ram_(animal)", "zebra", "seabird", "starfish"}
PRIORITY = list(FAMILIES)
# captions that describe flat graphics / text rather than forms
JUNK = re.compile(r"\b(kitchen|dew|bike|bicycle|terrain|landscape|tile|ground|floor|patch|text|logo|letter|letters|word|words|sign|card|poster|map|diagram|icon|emoji|texture|pattern|flag|painting|picture|photo|image|screen|number|alphabet)\b", re.I)
# things that break the mood: toys, plushies, cartoon/IP characters, voxel art
# (also drops existing library forms at export, via mimoid_quality.py)
MOOD = re.compile(r"\b(toy|toys|plush\w*|stuffed|teddy|lego|chibi|kawaii|funko|pok[eé]mon|pikachu|minecraft|smiley|"
                  r"cartoon\w*|doll|dolls|keychain|keyring|pinata|disney|pixar|anime|mcqueen|mario|sonic|spongebob|"
                  r"pixelated|my little pony|rainbow dash|no[- ]face|spirited away|five nights at freddy'?s|mickey mouse|"
                  r"transformers|ice cream|cakes?|cupcakes?|lollipops?|candy|sushi|skateboards?|kites?|soda|beer|can of|"
                  r"rain ?boots?|rubber boots?|gumboots?|rain ?jacket|raincoat)\b|(?<!hot air )(?<!hot-air )\bballoons?\b", re.I)

# `select --extend`: settings and atmosphere, so a scene thought's middle and
# shallow layers have somewhere to go ("tea in grandmas kitchen during a
# rainstorm": the storm had only clouded mountains). Appended to the manifest;
# the regex's matched keyword is round-robined so one keyword (5.8k staircases,
# 2k catalog lamps) can't fill a family. family -> (quota, regex, exclude)
EXTRA_FAMILIES = {
    "weather":    (600, r"\b(cloud|clouds|cloudy|storm|storms|thunderstorm|thundercloud|raincloud|rain|raindrop|raindrops|lightning|tornado|hurricane|cyclone|twister|fog|foggy|mist|misty|snowflake|snowflakes|snowstorm|blizzard|rainbow|icicle|icicles|frost)\b",
                   r"point cloud|rainbow[- ](colou?red|striped)|word cloud|cloud (server|computing|storage)"),
    "water":      (1200, r"\b(wave|waves|fountain|fountains|(?:wishing|water|stone|wooden|old) well|waterfall|waterfalls|whirlpool|splash|droplet|droplets|water ?drop|puddle|pond|lake|river|sea|ocean|anchor|buoy|pier|dock|jetty|reef|body of water)\b",
                   r"water (bottle|filter|heater|gun|tank|pump|cooler|tap|dispenser)|filtration|flosser|docking|dock station|sea ?shell"),
    "light":      (2000, r"\b(lantern|lanterns|candle|candles|candlestick|candelabra|chandelier|torch|torches|sconce|lamppost|lamp post|streetlight|street light|street lamp|oil lamp|stone lamp|beacon|brazier|moon|crescent|sun|planet|comet|light bulb|lightbulb)\b",
                   r"sun ?glasses|sunflower|flashlight|moon ?boot"),
    "ruins":      (1200, r"\b(ruin|ruins|ruined|abandoned|crumbling|broken|destroyed|wreck|wreckage|shipwreck|aqueduct|colosseum|stonehenge|tomb|crypt|mausoleum|sarcophagus|catacomb|dolmen|cairn|altar|archway|gateway|ancient)\b",
                   r"\b(coin|coins|phone|screen|glass|bottle)\b"),
    "rooms":      (2500, r"\b(room|rooms|bedroom|kitchen|interior|hallway|corridor|staircase|stairs|stairway|fireplace|hearth|doorway|door|window|armchair|rocking chair|throne|bed|wardrobe|bookcase|bookshelf|grandfather clock|piano|bathtub|stove)\b",
                   r"\b(car door|door handle|doorknob|window frame sticker|bed ?bug|flower ?bed)\b"),
    "landscapes": (2000, r"\b(landscape|terrain|valley|forest|woods|desert|dune|dunes|meadow|beach|coast|coastline|shore|fjord|archipelago|plateau|ridge|gorge|ravine|oasis|swamp|marsh|jungle|hillside|floating island)\b",
                   r"topographic|\bmap\b"),
}
# the base JUNK minus the setting words the extension is looking for
JUNK_EXTRA = re.compile(r"\b(dew|bike|bicycle|tile|patch|text|logo|letter|letters|word|words|sign|card|poster|map|diagram|icon|emoji|texture|pattern|flag|painting|picture|photo|image|screen|number|alphabet)\b", re.I)
ALL_FAMILIES = [*FAMILIES, *EXTRA_FAMILIES]


def subject(caption: str) -> str:
    """The caption's head noun phrase: 'A weathered stone bust with ...' -> 'A weathered stone bust'."""
    return re.split(r",| with | featuring | on | holding | wearing | and | in | of a| that ", caption, maxsplit=1)[0]


# ------------------------------------------------------------------- index
def fetch(url: str, timeout: float = 90.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "solaris-mimoid/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def index_shard(i: int) -> dict:
    raw = json.loads(gzip.decompress(fetch(f"{HF}metadata/000-{i:03d}.json.gz")))
    out = {}
    for uid, m in raw.items():
        glb = (m.get("archives") or {}).get("glb") or {}
        out[uid] = [m.get("license"), round((glb.get("size") or 0) / 1e6, 2), glb.get("faceCount") or m.get("faceCount") or 0,
                    m.get("name") or "", ((m.get("user") or {}).get("displayName") or ""), m.get("viewerUrl") or ""]
    return out


def stage_index(args):
    dst = OBJ_DIR / "meta_compact.json.gz"
    if dst.exists():
        print(f"{dst} exists"); return
    meta = {}
    with ThreadPoolExecutor(8) as ex:
        for n, d in enumerate(ex.map(index_shard, range(160))):
            meta.update(d)
            if (n + 1) % 20 == 0:
                print(f"  {n + 1}/160 shards · {len(meta)} objects", flush=True)
    with gzip.open(dst, "wt", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"index -> {dst} ({len(meta)} objects)")


# ------------------------------------------------------------------ select
def manifest_row(uid, fam, caption, path, m):
    return {"uid": uid, "family": fam, "caption": caption, "path": path, "license": m[0],
            "mb": m[1], "name": m[3], "author": m[4], "url": m[5]}


def stage_extend(args):
    """Append EXTRA_FAMILIES to an existing manifest (never touches its rows, so
    built shards, AE latents and quality rows keep their order)."""
    dst = OUT_DIR / args.manifest
    have = [json.loads(l) for l in open(dst, encoding="utf-8")]
    done_fams = {r["family"] for r in have} & set(EXTRA_FAMILIES)
    if done_fams:
        print(f"{dst.name} already has {sorted(done_fams)}; nothing to do"); return
    have_uids = {r["uid"] for r in have}
    meta = json.load(gzip.open(OBJ_DIR / "meta_compact.json.gz", "rt", encoding="utf-8"))
    paths = json.load(gzip.open(OBJ_DIR / "object-paths.json.gz"))
    regs = {k: re.compile(r, re.I) for k, (_, r, _) in EXTRA_FAMILIES.items()}
    excl = {k: re.compile(x, re.I) for k, (_, _, x) in EXTRA_FAMILIES.items()}
    # family -> keyword -> [(uid, caption)]
    cands = {k: {} for k in EXTRA_FAMILIES}
    seen = set()
    n_mood = 0
    with open(OBJ_DIR / "cap3d.csv", encoding="utf-8", errors="replace") as f:
        for uid, cap in csv.reader(f):
            if len(uid) != 32 or uid in have_uids or uid in seen:
                continue
            m = meta.get(uid)
            if not (m and uid in paths and m[0] in LICENSES and 0 < m[1] <= MAX_MB and m[2] <= MAX_FACES):
                continue
            seen.add(uid)
            subj = subject(cap)
            if JUNK_EXTRA.search(subj):
                continue
            for fam, rg in regs.items():
                hit = rg.search(subj)
                if hit and not excl[fam].search(cap):
                    if MOOD.search(cap):
                        n_mood += 1
                    else:
                        cands[fam].setdefault(hit.group(1).lower(), []).append((uid, cap))
                    break
    print(f"  dropped {n_mood} toy/cartoon matches")
    rows = []
    for fam, (quota, _, _) in EXTRA_FAMILIES.items():
        by_kw = {kw: sorted(v, key=lambda x: hashlib.sha1(f"mimoid{x[0]}".encode()).hexdigest())
                 for kw, v in sorted(cands[fam].items())}
        take, mix = [], {}
        depth = 0
        while len(take) < quota and any(depth < len(v) for v in by_kw.values()):
            for kw, v in by_kw.items():  # round-robin: every keyword's next form in turn
                if depth < len(v) and len(take) < quota:
                    take.append(v[depth])
                    mix[kw] = mix.get(kw, 0) + 1
            depth += 1
        n_pool = sum(len(v) for v in by_kw.values())
        print(f"  {fam:10s} {n_pool:6d} candidates -> {len(take):5d} ({sum(meta[u][1] for u, _ in take) / 1e3:.1f} GB) · "
              + ", ".join(f"{kw} {n}" for kw, n in sorted(mix.items(), key=lambda x: -x[1])[:8]))
        rows += [manifest_row(u, fam, cap, paths[u], meta[u]) for u, cap in take]
    backup = dst.with_name(dst.stem + "_base.jsonl")
    if not backup.exists():
        backup.write_bytes(dst.read_bytes())
    with open(dst, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"manifest += {len(rows)} objects -> {dst} ({sum(r['mb'] for r in rows) / 1e3:.1f} GB to stream; "
          f"original kept as {backup.name})")


def stage_select(args):
    if args.extend:
        return stage_extend(args)
    meta = json.load(gzip.open(OBJ_DIR / "meta_compact.json.gz", "rt", encoding="utf-8"))
    paths = json.load(gzip.open(OBJ_DIR / "object-paths.json.gz"))
    lvis = json.load(gzip.open(OBJ_DIR / "lvis-annotations.json.gz"))
    lvis_fam = {}
    for cat, uids in lvis.items():
        fam = LVIS_TO_FAMILY.get(cat) or ("animals" if cat in LVIS_ANIMALS else None)
        if fam:
            for u in uids:
                lvis_fam[u] = fam
    regs = {k: re.compile(r, re.I) for k, (_, r) in FAMILIES.items()}

    def ok(uid):
        m = meta.get(uid)
        return m and uid in paths and m[0] in LICENSES and 0 < m[1] <= MAX_MB and m[2] <= MAX_FACES

    cands = {k: [] for k in FAMILIES}
    captions = {}
    with open(OBJ_DIR / "cap3d.csv", encoding="utf-8", errors="replace") as f:
        for uid, cap in csv.reader(f):
            if len(uid) != 32 or not ok(uid):
                continue
            captions[uid] = cap
            subj = subject(cap)
            if JUNK.search(subj):
                continue
            fam = lvis_fam.get(uid) or next((k for k in PRIORITY if regs[k].search(subj)), None)
            if fam:
                cands[fam].append(uid)
    for uid, fam in lvis_fam.items():  # LVIS objects Cap3D missed
        if uid not in captions and ok(uid):
            captions[uid] = meta[uid][3]
            cands[fam].append(uid)

    scale = args.total / sum(q for q, _ in FAMILIES.values())
    rows = []
    for fam, (quota, _) in FAMILIES.items():
        pool = sorted(set(cands[fam]), key=lambda u: hashlib.sha1(f"mimoid{u}".encode()).hexdigest())
        take = pool[: max(1, round(quota * scale))]
        mb = sum(meta[u][1] for u in take)
        print(f"  {fam:10s} {len(pool):6d} candidates -> {len(take):5d} ({mb / 1e3:.1f} GB)")
        rows += [manifest_row(u, fam, captions[u], paths[u], meta[u]) for u in take]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / args.manifest
    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"manifest -> {dst} ({len(rows)} objects, {sum(r['mb'] for r in rows) / 1e3:.1f} GB to stream)")


# ------------------------------------------------------------------- build
def mesh_to_grid(data: bytes):
    """GLB bytes -> int8 [G,G,G] truncated SDF, or (None, reason)."""
    import trimesh
    from scipy import ndimage

    try:
        mesh = trimesh.load(io.BytesIO(data), file_type="glb", force="mesh", process=False)
    except Exception as e:  # noqa: BLE001 — any unreadable file is just skipped
        return None, f"load: {type(e).__name__}"
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return None, "empty"
    v = np.asarray(mesh.vertices, dtype=np.float64)
    lo, hi = v.min(0), v.max(0)
    half = (hi - lo).max() / 2
    if not np.isfinite(half) or half <= 0:
        return None, "degenerate"
    mesh.vertices = (v - (lo + hi) / 2) * (EXTENT / half)
    area = float(mesh.area)
    if area <= 1e-6:
        return None, "zero-area"

    # unsigned distance: splat dense surface samples into a fine voxel grid and
    # run an exact Euclidean distance transform (linear time; a KD-tree query
    # of millions of grid points is ~100x slower here)
    n_pts = int(np.clip(area / (0.35 * H_UDF) ** 2, 200_000, 1_500_000))
    pts, _ = trimesh.sample.sample_surface(mesh, n_pts, seed=0)
    ijk = np.clip(np.round((pts + 1) / H_UDF).astype(np.int32), 0, G_UDF - 1)
    surf = np.zeros((G_UDF,) * 3, dtype=bool)
    surf[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    udf = ndimage.distance_transform_edt(~surf).astype(np.float32) * H_UDF

    # sign by flood fill on the coarser G_FINE grid: everything reachable from
    # the border without crossing the surface band is outside; enclosed space
    # (and the band) is solid — seals mesh cracks narrower than ~1 fine voxel
    udf_f = udf[::2, ::2, ::2]
    free = udf_f > 0.87 * H_FINE
    lab, _ = ndimage.label(free)
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
                                       lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    outside = np.isin(lab[::2, ::2, ::2], border[border > 0])
    sdf = np.where(outside, udf[::4, ::4, ::4], -udf[::4, ::4, ::4]) - OFFSET
    mask = sdf < 0
    solid = float(mask.mean())
    if solid < 0.002:
        return None, f"too-thin {solid:.4f}"
    if solid > 0.6:
        return None, f"too-full {solid:.2f}"
    # plates/cards/map tiles: paper-thin along some axis
    ext = np.array([mask.any(axis=tuple(a for a in range(3) if a != k)).sum() for k in range(3)])
    if ext.min() < 0.12 * ext.max():
        return None, f"flat {ext.min() / ext.max():.2f}"
    # scattered debris / multi-object scenes: no dominant connected body
    lab, _ = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())[1:]
    if sizes.max() < 0.6 * sizes.sum():
        return None, f"scattered {sizes.max() / sizes.sum():.2f}"
    return np.round(np.clip(sdf / TRUNC, -1, 1) * 127).astype(np.int8), f"ok {solid:.3f}"


def build_one(row):
    t0 = time.time()
    try:
        data = fetch(HF + row["path"])
    except Exception as e:  # noqa: BLE001
        return row["uid"], None, f"download: {type(e).__name__}", 0.0, 0.0
    t1 = time.time()
    try:
        grid, why = mesh_to_grid(data)
    except MemoryError:  # transient (other workers hold big grids too): retried on the next run
        grid, why = None, "oom"
    except Exception as e:  # noqa: BLE001 — one bad mesh must not stop the build
        grid, why = None, f"error: {type(e).__name__}"
    return row["uid"], grid, why, t1 - t0, time.time() - t1


def stage_build(args):
    rows = [json.loads(l) for l in open(OUT_DIR / args.manifest, encoding="utf-8")]
    tag = Path(args.manifest).stem
    log_path = OUT_DIR / f"{tag}.log.jsonl"
    done = set()
    if log_path.exists():
        done = {json.loads(l)["uid"] for l in open(log_path, encoding="utf-8")}
    todo = [r for r in rows if r["uid"] not in done]
    if args.limit:  # the next N unbuilt rows (a smoke test of an extended manifest)
        todo = todo[: args.limit]
    shard_idx = len(list(OUT_DIR.glob(f"{tag}_*.npz")))
    print(f"{len(rows)} in manifest · {len(done)} already done · {len(todo)} to build · {args.workers} workers", flush=True)

    buf_g, buf_u = [], []
    def flush():
        nonlocal shard_idx, buf_g, buf_u
        if not buf_g:
            return
        np.savez_compressed(OUT_DIR / f"{tag}_{shard_idx:04d}.npz", grids=np.stack(buf_g), uids=np.array(buf_u))
        shard_idx += 1
        buf_g, buf_u = [], []
        # log lines for the shard's objects are only trusted once the shard exists
        with open(log_path, "a", encoding="utf-8") as f:
            for rec in pending_log:
                f.write(json.dumps(rec) + "\n")
        pending_log.clear()

    pending_log = []
    t0 = time.time()
    n_ok = n_fail = n_retry = 0
    dl_s = cv_s = 0.0
    mb = 0.0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(build_one, r): r for r in todo}
        try:
            for k, fut in enumerate(as_completed(futs)):
                row = futs[fut]
                try:
                    uid, grid, why, dt_dl, dt_cv = fut.result()
                except BrokenProcessPool:
                    raise  # a worker died: the rest can't run; buffered grids are saved below
                except Exception as e:  # noqa: BLE001 — e.g. a result too big to pickle back
                    uid, grid, why, dt_dl, dt_cv = row["uid"], None, f"error: {type(e).__name__}", 0.0, 0.0
                dl_s += dt_dl; cv_s += dt_cv; mb += row["mb"]
                if why == "oom":  # not logged, so a rerun tries it again
                    n_retry += 1
                else:
                    pending_log.append({"uid": uid, "status": why})
                if grid is not None:
                    buf_g.append(grid); buf_u.append(uid); n_ok += 1
                else:
                    n_fail += 1
                if len(buf_g) >= SHARD:
                    flush()
                if (k + 1) % 25 == 0 or k + 1 == len(todo):
                    el = time.time() - t0
                    eta = el / (k + 1) * (len(todo) - k - 1) / 3600
                    print(f"  {k + 1}/{len(todo)} · ok {n_ok} fail {n_fail} · {mb / el:.1f} MB/s · "
                          f"avg dl {dl_s / (k + 1):.1f}s conv {cv_s / (k + 1):.1f}s · eta {eta:.1f} h", flush=True)
        finally:
            flush()
    # failures carry no grid, so they may still be pending if the last shard was empty
    if pending_log:
        with open(log_path, "a", encoding="utf-8") as f:
            for rec in pending_log:
                f.write(json.dumps(rec) + "\n")
    print(f"build done: {n_ok} grids, {n_fail} skipped ({n_retry} out of memory, retried next run), "
          f"{(time.time() - t0) / 60:.1f} min")


# ----------------------------------------------------------------- preview
def render_grids(grids: np.ndarray, size: int = 160, yaw: float = 0.6, pitch: float = 0.25):
    """Sphere-trace a batch of int8 SDF grids on the GPU -> uint8 [B, size, size, 3].
    Same trilinear data path as the browser raymarcher."""
    import torch
    import torch.nn.functional as F

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vol = torch.from_numpy(grids.astype(np.float32) / 127.0 * TRUNC).to(dev)[:, None]  # [B,1,X,Y,Z]
    B = vol.shape[0]
    cam = 3.2
    eye = torch.tensor([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)], dtype=torch.float32, device=dev) * cam
    fwd = -eye / eye.norm()
    right = torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=dev)); right /= right.norm()
    up = torch.linalg.cross(right, fwd)
    f = np.tan(np.radians(36) / 2)
    xs = (torch.arange(size, device=dev) + 0.5) / size * 2 - 1
    gy, gx = torch.meshgrid(xs, xs, indexing="ij")
    rd = fwd + gx.reshape(-1, 1) * f * right - gy.reshape(-1, 1) * f * up
    rd = rd / rd.norm(dim=1, keepdim=True)
    R = rd.shape[0]

    def sample(p):  # p [B,R,3] in [-1,1] (x,y,z) -> [B,R]
        g = p[..., [2, 1, 0]].reshape(B, R, 1, 1, 3)  # grid_sample wants (W,H,D) = (z,y,x)
        d = F.grid_sample(vol, g, mode="bilinear", padding_mode="border", align_corners=True).reshape(B, R)
        return d + (p.abs() - 1).clamp(min=0).norm(dim=-1)  # beyond the cube: border + distance to it

    # start rays at the bounding cube
    t = torch.full((B, R), cam - 1.8, device=dev)
    hit = torch.zeros(B, R, dtype=torch.bool, device=dev)
    for _ in range(160):
        p = eye + rd[None] * t[..., None]
        d = sample(p)
        hit |= d < 1e-3
        t = torch.where(hit, t, t + d.clamp(min=2e-3) * 0.9)
    p = eye + rd[None] * t[..., None]
    hit &= t < cam + 1.8
    e = 1.5 / (G - 1)
    n = torch.stack([sample(p + e * torch.eye(3, device=dev)[k]) - sample(p - e * torch.eye(3, device=dev)[k]) for k in range(3)], -1)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    l1 = torch.tensor([0.6, 0.8, 0.5], device=dev); l1 /= l1.norm()
    l2 = torch.tensor([-0.7, -0.2, -0.4], device=dev); l2 /= l2.norm()
    dif = (n @ l1).clamp(0, 1) * 0.9 + (n @ l2).clamp(0, 1) * 0.3
    fres = (1 - (n * -rd[None]).sum(-1).clamp(0, 1)) ** 3
    base = torch.tensor([0.35, 0.55, 0.62], device=dev)
    col = base * (0.08 + dif[..., None]) + fres[..., None] * (0.5 * base + 0.3 * torch.tensor([0.9, 0.5, 0.8], device=dev))
    col = 1 - torch.exp(-col * 1.8)
    bg = torch.tensor([0.016, 0.012, 0.024], device=dev)
    img = torch.where(hit[..., None], col, bg)
    img = (img.clamp(0, 1) ** (1 / 1.6) * 255).byte().reshape(B, size, size, 3)
    return img.cpu().numpy()


def stage_preview(args):
    sys.path.insert(0, str(ROOT / "scripts"))
    from imagination_corpus import write_png

    from mimoid_train_ae import load_all  # the memmapped cache: the full library doesn't fit in RAM twice

    tag = Path(args.manifest).stem
    rows = {json.loads(l)["uid"]: json.loads(l) for l in open(OUT_DIR / args.manifest, encoding="utf-8")}
    grids, uids = load_all(tag)
    print(f"{len(uids)} grids in library")
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fam_of = np.array([rows[u]["family"] for u in uids])
    for fam in args.families.split(",") if args.families else ALL_FAMILIES:
        idx = np.where(fam_of == fam)[0][: args.per_family]
        if len(idx) == 0:
            continue
        tiles = render_grids(np.asarray(grids[idx]))
        cols = 6
        pad = (-len(tiles)) % cols
        tiles = np.concatenate([tiles, np.zeros((pad,) + tiles.shape[1:], np.uint8)])
        sheet = tiles.reshape(-1, cols, *tiles.shape[1:]).transpose(0, 2, 1, 3, 4).reshape(-1, cols * tiles.shape[2], 3)
        path = PREVIEW_DIR / f"{tag}_{fam}.png"
        write_png(path, np.ascontiguousarray(sheet))
        with open(PREVIEW_DIR / f"{tag}_{fam}.txt", "w", encoding="utf-8") as f:
            for i in idx:
                f.write(f"{uids[i]}  {rows[uids[i]]['caption'][:110]}\n")
        print(f"  {fam}: {len(idx)} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["index", "select", "build", "preview"])
    ap.add_argument("--total", type=int, default=20500, help="select: library size")
    ap.add_argument("--extend", action="store_true", help="select: append EXTRA_FAMILIES to the existing manifest")
    ap.add_argument("--manifest", default="library.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="build: only the next N unbuilt rows")
    ap.add_argument("--families", default="", help="preview: comma-separated families (default all)")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--per-family", type=int, default=18)
    args = ap.parse_args()
    {"index": stage_index, "select": stage_select, "build": stage_build, "preview": stage_preview}[args.stage](args)


if __name__ == "__main__":
    main()
