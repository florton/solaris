// The tour: thoughts the piece dreams on its own after loading, in order,
// looping. Re-picked for the 4D walk (scripts/mimoid_dream.py; every layer
// from one neighbourhood) from a rendered sweep of candidates (scratchpad
// preset_candidates.py) for forms that hold up on every layer; several walk
// through different objects across the layers (elephant, chess piece, whale).
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
];

export const TOUR = {
  cloudMs: 3000, // the thought's cloud shows at least this long before it condenses
  holdMs: 22000, // how long a condensed thought stays before the next one
  idleMs: 6000, // never advance while someone is orbiting/scrolling
  resumeMs: 60000, // after a typed thought, rejoin the tour once idle this long
};
