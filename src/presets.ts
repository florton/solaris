// The tour: thoughts the piece dreams on its own after loading, in order,
// looping from a random start. Re-picked for the 4D walk
// (scripts/mimoid_dream.py; every layer from one neighbourhood) from a
// rendered sweep of candidates (scratchpad preset_candidates.py) for forms
// that hold up on every layer; several walk through different objects across
// the layers (elephant, chess piece, whale).
export const PRESETS = [
  'an elephant walking through a dream',
  'a stone pillar in the desert',
  'an owl watching from a branch',
  'a chess piece the size of a building',
  'a whale drifting through fog',
  'the ruins of a castle tower',
  'a church bell ringing at noon',
  'the statue of a forgotten king',
  'a small house at the edge of the world',
  'the bust of a woman in marble',
  'an old lighthouse on a cliff',
  'a skull in the sand',
  'a sailing ship lost at sea',
  'a dragon asleep on its gold',
  // v0.04: scenes that reach the grown families (weather, light, ruins, rooms,
  // landscapes) on the shallow layers; scratchpad preset_more.py
  'a lantern burning in the dark',
  'a thunderstorm over the mountains',
  'a broken arch in the ruins of rome',
  'a horse standing in the rain',
  'a candle in an empty room',
  'a stone bridge over a river',
  'the face of an old man carved in stone',
  'a windmill on a hill',
  'an old armchair by the fireplace',
  'a deer in a winter forest',
  'a vintage car parked under a streetlight',
  'a crystal growing in a cave',
  'an ancient temple swallowed by the jungle',
  'a grand piano in an empty hall',
  'a lion resting on a rock',
  'a volcano at the edge of the world',
];

export const TOUR = {
  cloudMs: 3000, // the thought's cloud shows at least this long before it condenses
  holdMs: 34000, // how long a condensed thought stays: one walk 6 -> 0 -> 6 at layerMs
  layerMs: 2800, // idle walk: time per layer (melt there, then rest on the whole form)
  walkTilt: 0.35, // idle walk: how far the slice leans into w while melting (rad)
  snapMs: 350, // a scroll settles onto the nearest layer after this long
  idleMs: 4000, // never advance while someone is orbiting/scrolling
  resumeMs: 60000, // after a typed thought, rejoin the tour once idle this long
};
