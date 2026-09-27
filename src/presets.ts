// The tour: thoughts the piece dreams on its own after loading, in order,
// looping. Picked from a rendered sweep of candidates (scratchpad
// preset_candidates.py) for forms that hold up on every layer; several walk
// through different objects across the layers (elephant, chess piece, whale).
export const PRESETS = [
  'an elephant walking through a dream',
  'a stone pillar in the desert',
  'the face of a sleeping giant',
  'a chess piece the size of a building',
  'a whale drifting through fog',
  'the ruins of a castle tower',
  'cathedral light on dust',
  'the statue of a forgotten king',
  'a small house at the edge of the world',
  'the bust of a woman in marble',
  'an old lighthouse on a cliff',
  'a skull in the sand',
];

export const TOUR = {
  cloudMs: 3000, // the thought's cloud shows at least this long before it condenses
  holdMs: 22000, // how long a condensed thought stays before the next one
  idleMs: 6000, // never advance while someone is orbiting/scrolling
  resumeMs: 60000, // after a typed thought, rejoin the tour once idle this long
};
