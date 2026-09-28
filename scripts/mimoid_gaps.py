# Where the library is thin: for settings/atmosphere phrases a scene thought
# splits into, the best kept library caption and its cosine (sentence
# embedding, as the browser retrieves). Below ~0.5 the phrase's layers are
# loose associations ("a storm" -> a roaring lion). Run before and after
# growing the library to compare.
# Run: .venv/Scripts/python scripts/mimoid_gaps.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mimoid_captions import embed  # noqa: E402
from mimoid_dream import CAPS, EMB_LAYER  # noqa: E402
from mimoid_quality import OUT as QUALITY, keep  # noqa: E402

PHRASES = {
    "weather": ["a rainstorm", "a storm", "fog", "a thunderstorm", "lightning", "clouds", "a tornado", "snow falling",
                "mist over the sea", "a hurricane", "rain"],
    "rooms": ["grandma's kitchen", "an empty room", "a bedroom", "a staircase", "a fireplace", "a hallway",
              "an old armchair", "a doorway", "a window", "a library"],
    "landscapes": ["a valley", "a forest", "a desert", "a waterfall", "a canyon", "dunes", "a meadow", "a coastline",
                   "a lake", "a mountain range"],
    "ruins": ["ancient ruins", "a ruined temple", "a crumbling wall", "an abandoned house", "a broken column",
              "a shipwreck", "a tomb", "an aqueduct"],
    "light": ["a lantern", "candlelight", "a chandelier", "a street lamp", "the moon", "the sun", "a lamp", "a torch",
              "cathedral light", "stars"],
    "water": ["the sea", "a wave", "a fountain", "a well", "a river", "an ocean", "a pond", "a whirlpool",
              "a drop of water", "an anchor"],
}


def main():
    caps = np.load(CAPS)
    k = keep(dict(np.load(QUALITY)))
    phrases = [p for v in PHRASES.values() for p in v]
    s = embed(phrases)[:, EMB_LAYER] @ caps["emb"][:, EMB_LAYER].astype(np.float32).T
    s[:, ~k] = -1
    print(f"{k.sum()} kept forms")
    i = 0
    for fam, ps in PHRASES.items():
        best = s[i:i + len(ps)].max(1)
        print(f"== {fam}  mean best {best.mean():.3f}")
        for p in ps:
            top = np.argsort(-s[i])[:3]
            print(f"  {s[i, top[0]]:.3f}  {p:20s} -> " + " | ".join(str(caps['captions'][t])[:40] for t in top))
            i += 1


if __name__ == "__main__":
    main()
