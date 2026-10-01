/* Original EvoPolis pixel art. No external sprites or image assets.
 * The map is presentation only: distance, homes and movement add no game rules.
 * Transfer flowers are symbolic cues, never a count of resource units.
 */

export const TOWN_WIDTH = 480;
export const TOWN_HEIGHT = 240;

export const residentPositions = Object.freeze([
  {x: 129, y: 91, homeX: 77, homeY: 79},
  {x: 351, y: 91, homeX: 404, homeY: 79},
  {x: 129, y: 184, homeX: 77, homeY: 174},
  {x: 351, y: 184, homeX: 404, homeY: 174},
]);

const P = {
  ink: "#172038", indigo: "#253a5e", slate: "#3b5278",
  cream: "#fff1d2", pale: "#b6c6df", cyan: "#6dcff6",
  mint: "#73e0b3", amber: "#ffcf6e", coral: "#ff8b8b",
  grass: "#4d7864", grassDark: "#426b5a", grassLight: "#628d70",
  treeDark: "#253f45", tree: "#345948", leaf: "#477953",
  leafLight: "#6c9864", soil: "#554943", path: "#a69b79",
  pathDark: "#847b65", pathLight: "#c7b995", wall: "#d6c8a2",
  wood: "#87624f", woodLight: "#b48a63",
};

// Identity colors intentionally differ from the mechanism chart encoding.
const residents = [
  {coat: "#ae8acb", light: "#d8bce6", roof: "#806293", hair: "#454352", skin: "#e4bf9b"},
  {coat: "#bbc9d7", light: "#e7edf0", roof: "#657d99", hair: "#866449", skin: "#b88c6e"},
  {coat: "#d59360", light: "#ebbc8f", roof: "#a3674c", hair: "#544849", skin: "#d3a985"},
  {coat: "#ceba8c", light: "#eee0b2", roof: "#8b8061", hair: "#565469", skin: "#ae7e62"},
];

let backdrop;

function rect(ctx, x, y, width, height, color) {
  ctx.fillStyle = color;
  ctx.fillRect(Math.round(x), Math.round(y), width, height);
}

function flower(ctx, x, y, color, stem = true) {
  if (stem) {
    rect(ctx, x, y + 2, 1, 5, P.treeDark);
    rect(ctx, x + 1, y + 4, 2, 1, P.leafLight);
  }
  rect(ctx, x - 1, y - 2, 3, 5, color);
  rect(ctx, x - 2, y - 1, 5, 3, color);
  rect(ctx, x, y, 1, 1, P.cream);
}

function tree(ctx, x, y, variant = 0) {
  const offset = variant % 2 ? 2 : 0;
  rect(ctx, x - 12, y + 23, 29, 5, P.grassDark);
  rect(ctx, x - 2, y + 8, 5, 17, P.ink);
  rect(ctx, x - 1, y + 9, 2, 14, P.wood);
  rect(ctx, x - 10, y - 6, 21, 26, P.treeDark);
  rect(ctx, x - 15, y, 31, 15, P.treeDark);
  rect(ctx, x - 6, y - 10, 14, 7, P.treeDark);
  rect(ctx, x - 12, y + 1, 25, 12, P.tree);
  rect(ctx, x - 8, y - 5, 20, 16, P.leaf);
  rect(ctx, x - 4, y - 8, 11, 6, P.leaf);
  rect(ctx, x - 8, y - 3, 6, 4, P.leafLight);
  rect(ctx, x - 4, y - 6, 5, 3, P.leafLight);
  rect(ctx, x + 2 + offset, y + 2, 6, 3, P.leafLight);
  rect(ctx, x - 12, y + 5, 6, 4, P.leaf);
  rect(ctx, x + 5, y + 11, 6, 3, P.treeDark);
  rect(ctx, x - 5, y + 9, 4, 3, P.tree);
}

function shrub(ctx, x, y) {
  rect(ctx, x - 1, y + 2, 13, 8, P.treeDark);
  rect(ctx, x + 2, y - 1, 7, 12, P.treeDark);
  rect(ctx, x + 1, y + 2, 9, 5, P.leaf);
  rect(ctx, x + 3, y, 5, 4, P.leafLight);
  rect(ctx, x + 7, y + 3, 2, 2, P.grassLight);
}

function fence(ctx, x, y, length) {
  rect(ctx, x, y + 2, length, 2, P.ink);
  rect(ctx, x, y + 6, length, 2, P.wood);
  for (let i = 0; i <= length; i += 10) {
    rect(ctx, x + i, y, 3, 11, P.ink);
    rect(ctx, x + i, y, 2, 9, P.woodLight);
    rect(ctx, x + i, y, 2, 2, P.pathLight);
  }
}

function house(ctx, x, y, resident, variant) {
  const r = residents[resident];
  rect(ctx, x - 4, y + 48, 69, 5, P.grassDark);
  rect(ctx, x + 1, y + 11, 58, 40, P.ink);
  rect(ctx, x + 4, y + 15, 52, 32, P.wall);
  rect(ctx, x + 4, y + 15, 52, 5, P.wood);
  rect(ctx, x + 4, y + 20, 3, 27, P.pathLight);
  rect(ctx, x + 51, y + 20, 5, 27, P.pathDark);
  rect(ctx, x - 4, y + 11, 68, 6, P.ink);
  rect(ctx, x, y + 7, 60, 7, P.ink);
  rect(ctx, x + 5, y + 3, 50, 8, P.ink);
  rect(ctx, x + 10, y - 1, 40, 8, P.ink);
  rect(ctx, x + 15, y - 5, 30, 8, P.ink);
  rect(ctx, x - 1, y + 10, 62, 4, r.roof);
  rect(ctx, x + 3, y + 6, 54, 5, r.roof);
  rect(ctx, x + 8, y + 2, 44, 5, r.roof);
  rect(ctx, x + 13, y - 2, 34, 5, r.roof);
  rect(ctx, x + 16, y - 4, 28, 2, r.light);
  for (let row = 0; row < 3; row++) {
    for (let tile = 0; tile < 5 - row; tile++) {
      rect(ctx, x + 7 + row * 5 + tile * 9, y + 10 - row * 4, 6, 1, r.coat);
    }
  }
  rect(ctx, x + 44, y - 10, 7, 17, P.ink);
  rect(ctx, x + 45, y - 9, 5, 12, P.pathDark);
  rect(ctx, x + 44, y - 10, 7, 3, P.pathLight);
  rect(ctx, x + 44, y - 4, 5, 1, P.wood);
  // Homes share geometry; roof and sprite identify A–D.
  [11, 39].forEach(wx => {
    rect(ctx, x + wx, y + 25, 11, 12, P.ink);
    rect(ctx, x + wx + 1, y + 26, 9, 9, P.pale);
    rect(ctx, x + wx + 2, y + 27, 3, 4, P.cream);
    rect(ctx, x + wx + 5, y + 26, 1, 9, P.slate);
    rect(ctx, x + wx + 1, y + 31, 9, 1, P.slate);
    rect(ctx, x + wx - 1, y + 36, 13, 2, P.wood);
  });
  rect(ctx, x + 26, y + 28, 11, 19, P.ink);
  rect(ctx, x + 28, y + 30, 7, 17, P.wood);
  rect(ctx, x + 28, y + 30, 1, 15, P.woodLight);
  rect(ctx, x + 32, y + 38, 2, 2, P.cream);
  rect(ctx, x + 24, y + 47, 15, 2, P.pathLight);
  rect(ctx, x + 22, y + 49, 19, 2, P.pathDark);
  if (variant) {
    rect(ctx, x + 9, y + 39, 13, 4, P.wood);
    flower(ctx, x + 12, y + 38, r.light, false);
    flower(ctx, x + 18, y + 39, r.coat, false);
  }
}

function hall(ctx) {
  const x = 213, y = 17;
  rect(ctx, x - 7, y + 48, 70, 6, P.grassDark);
  rect(ctx, x, y + 12, 56, 38, P.ink);
  rect(ctx, x + 3, y + 16, 50, 31, P.wall);
  rect(ctx, x + 7, y + 18, 4, 29, P.pathLight);
  rect(ctx, x + 46, y + 18, 4, 29, P.pathDark);
  rect(ctx, x - 7, y + 11, 70, 6, P.ink);
  rect(ctx, x - 3, y + 7, 62, 7, P.ink);
  rect(ctx, x + 2, y + 3, 52, 8, P.ink);
  rect(ctx, x + 8, y - 1, 40, 8, P.ink);
  rect(ctx, x + 14, y - 5, 28, 8, P.ink);
  rect(ctx, x - 4, y + 11, 64, 3, P.slate);
  rect(ctx, x, y + 7, 56, 4, P.indigo);
  rect(ctx, x + 5, y + 3, 46, 4, P.slate);
  rect(ctx, x + 11, y - 1, 34, 4, P.indigo);
  rect(ctx, x + 17, y - 4, 22, 3, P.slate);
  rect(ctx, x + 19, y + 3, 18, 14, P.ink);
  rect(ctx, x + 21, y + 4, 14, 11, P.wall);
  flower(ctx, x + 28, y + 9, P.wood, false);
  rect(ctx, x + 21, y + 29, 16, 18, P.ink);
  rect(ctx, x + 23, y + 31, 12, 16, P.wood);
  rect(ctx, x + 28, y + 31, 2, 16, P.ink);
  rect(ctx, x + 25, y + 38, 1, 2, P.cream);
  rect(ctx, x + 32, y + 38, 1, 2, P.cream);
  [12, 40].forEach(wx => {
    rect(ctx, x + wx, y + 24, 6, 10, P.ink);
    rect(ctx, x + wx + 1, y + 25, 4, 7, P.pale);
    rect(ctx, x + wx + 1, y + 25, 2, 3, P.cream);
  });
  rect(ctx, x + 17, y + 47, 24, 3, P.pathLight);
  rect(ctx, x + 14, y + 50, 30, 2, P.pathDark);
}

function bench(ctx, x, y) {
  rect(ctx, x + 2, y + 6, 2, 6, P.ink);
  rect(ctx, x + 17, y + 6, 2, 6, P.ink);
  rect(ctx, x, y, 22, 3, P.ink);
  rect(ctx, x + 1, y, 20, 2, P.woodLight);
  rect(ctx, x, y + 4, 22, 4, P.ink);
  rect(ctx, x + 1, y + 4, 20, 2, P.woodLight);
}

function lantern(ctx, x, y) {
  rect(ctx, x + 1, y + 2, 2, 15, P.ink);
  rect(ctx, x - 3, y, 9, 2, P.ink);
  rect(ctx, x - 2, y - 7, 7, 7, P.ink);
  rect(ctx, x - 1, y - 6, 5, 5, P.pathLight);
  rect(ctx, x, y - 5, 2, 4, P.cream);
  rect(ctx, x - 1, y - 9, 5, 2, P.ink);
  rect(ctx, x - 1, y + 15, 6, 2, P.treeDark);
}

function buildBackdrop() {
  const canvas = document.createElement("canvas");
  canvas.width = TOWN_WIDTH;
  canvas.height = TOWN_HEIGHT;
  const ctx = canvas.getContext("2d");
  rect(ctx, 0, 0, 480, 240, P.grass);
  // A deterministic tile texture; redraws and seeking cannot change scenery.
  for (let y = 4; y < 240; y += 8) {
    for (let x = 3; x < 480; x += 11) {
      const seed = (x * 73 + y * 29) % 19;
      if (seed < 6) {
        rect(ctx, x + seed, y, 2, 1, P.grassDark);
        rect(ctx, x + seed + 2, y + 1, 1, 2, P.grassDark);
      } else if (seed === 12) {
        rect(ctx, x, y, 2, 1, P.grassLight);
      }
    }
  }

  // One shared crossroad. All four transfers occur simultaneously.
  const paths = [[66, 82, 350, 16], [66, 177, 350, 16], [228, 63, 24, 143],
    [68, 68, 17, 24], [395, 68, 17, 24], [68, 163, 17, 23], [395, 163, 17, 23],
    [185, 93, 109, 85]];
  paths.forEach(([x, y, w, h]) => {
    rect(ctx, x - 2, y - 2, w + 4, h + 4, P.pathDark);
    rect(ctx, x, y, w, h, P.path);
  });
  for (let y = 65; y < 201; y += 9) {
    for (let x = 69; x < 415; x += 13) {
      if (paths.some(([px, py, w, h]) => x >= px + 2 && x < px + w - 5 && y >= py + 2 && y < py + h - 2)) {
        if ((x + y) % 3 === 0) rect(ctx, x, y, 4, 1, P.pathLight);
        else rect(ctx, x, y, 2, 1, P.pathDark);
      }
    }
  }

  // Low orchard walls and planting plots give the four homes a common setting.
  fence(ctx, 44, 108, 61);
  fence(ctx, 374, 108, 61);
  fence(ctx, 44, 205, 61);
  fence(ctx, 374, 205, 61);
  [[147, 37], [305, 37], [144, 210], [305, 210]].forEach(([x, y], i) => {
    rect(ctx, x, y, 24, 13, P.treeDark);
    rect(ctx, x + 1, y + 1, 22, 10, P.soil);
    for (let j = 0; j < 4; j++) {
      flower(ctx, x + 4 + j * 5, y + 3 + (j % 2) * 3, residents[i].light);
    }
  });
  [[17, 20], [39, 6], [118, 9], [168, 11], [193, 8], [292, 9], [318, 8], [368, 7], [443, 6], [466, 19],
    [14, 55], [15, 106], [465, 60], [468, 103], [17, 144], [466, 146],
    [15, 198], [18, 230], [46, 232], [118, 230], [177, 234], [301, 234], [362, 232], [439, 232], [466, 202], [468, 231]]
    .forEach(([x, y], i) => tree(ctx, x, y, i));
  [[112, 48], [108, 140], [360, 49], [361, 140], [163, 127], [305, 126], [189, 195], [284, 195],
    [58, 117], [406, 118], [219, 218], [251, 219]].forEach(([x, y]) => shrub(ctx, x, y));
  house(ctx, 45, 28, 0, true);
  house(ctx, 373, 28, 1, false);
  house(ctx, 45, 123, 2, false);
  house(ctx, 373, 123, 3, true);
  hall(ctx);
  bench(ctx, 154, 66);
  bench(ctx, 305, 66);
  bench(ctx, 198, 204);
  bench(ctx, 260, 204);
  lantern(ctx, 175, 164);
  lantern(ctx, 306, 164);
  // A little stone well is scenery, separate from the numerical commons.
  rect(ctx, 149, 152, 17, 10, P.ink);
  rect(ctx, 151, 153, 13, 7, P.slate);
  rect(ctx, 153, 153, 9, 3, P.ink);
  rect(ctx, 149, 150, 17, 3, P.pale);
  rect(ctx, 149, 140, 2, 14, P.wood);
  rect(ctx, 164, 140, 2, 14, P.wood);
  rect(ctx, 148, 138, 19, 3, P.ink);
  rect(ctx, 150, 136, 15, 3, P.woodLight);
  return canvas;
}

function commons(ctx, pool, phase, missingAfter) {
  rect(ctx, 196, 109, 88, 50, P.ink);
  rect(ctx, 198, 111, 84, 46, P.slate);
  rect(ctx, 200, 113, 80, 42, P.pale);
  rect(ctx, 203, 116, 74, 34, P.ink);
  rect(ctx, 205, 118, 70, 30, P.soil);
  rect(ctx, 200, 151, 80, 4, P.pathDark);
  for (let x = 206; x < 279; x += 12) rect(ctx, x, 152, 1, 3, P.slate);

  // Flower density is a schematic stock cue. The interface carries exact values.
  // A positive recorded 0.01 pool retains one bloom; it is never silently zeroed.
  const flowerCount = typeof pool === "number" && Number.isFinite(pool)
    ? Math.min(28, Math.max(0, Math.ceil(pool / 200 * 28))) : 0;
  for (let i = 0; i < 28; i++) {
    const x = 210 + (i % 7) * 9;
    const y = 121 + Math.floor(i / 7) * 7;
    rect(ctx, x - 1, y + 3, 3, 1, P.wood);
    if (i < flowerCount) flower(ctx, x, y, i % 4 === 0 ? P.cream : P.mint);
  }
  // Square sign identifies the commons without suggesting invented rules.
  rect(ctx, 211, 99, 58, 13, P.ink);
  rect(ctx, 212, 100, 56, 11, P.indigo);
  ctx.fillStyle = P.cream;
  ctx.font = "bold 8px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("COMMONS", 240, 106);
  if (phase === 2) {
    const color = missingAfter ? P.pale : P.mint;
    [[193, 106, 7, 2], [193, 106, 2, 7], [280, 106, 7, 2], [285, 106, 2, 7],
      [193, 160, 7, 2], [193, 155, 2, 7], [280, 160, 7, 2], [285, 155, 2, 7]]
      .forEach(([x, y, w, h]) => rect(ctx, x, y, w, h, color));
    if (missingAfter) {
      rect(ctx, 232, 126, 16, 16, P.ink);
      ctx.fillStyle = P.cream;
      ctx.font = "bold 12px monospace";
      ctx.fillText("?", 240, 134);
    }
  }
}

function resident(ctx, index, selected, limited) {
  const {x, y} = residentPositions[index];
  const r = residents[index];
  rect(ctx, x - 8, y + 6, 17, 3, P.pathDark);
  if (selected) {
    [[x - 11, y - 20, 7, 2], [x - 11, y - 20, 2, 7], [x + 5, y - 20, 7, 2], [x + 10, y - 20, 2, 7],
      [x - 11, y + 10, 7, 2], [x - 11, y + 5, 2, 7], [x + 5, y + 10, 7, 2], [x + 10, y + 5, 2, 7]]
      .forEach(([rx, ry, w, h]) => rect(ctx, rx, ry, w, h, P.cream));
  }
  // All residents remain present even when the current offer is below one.
  rect(ctx, x - 4, y + 1, 4, 7, P.ink);
  rect(ctx, x + 2, y + 1, 4, 7, P.ink);
  rect(ctx, x - 4, y + 1, 3, 4, P.slate);
  rect(ctx, x + 2, y + 1, 3, 4, P.slate);
  rect(ctx, x - 6, y - 7, 13, 10, P.ink);
  rect(ctx, x - 5, y - 7, 11, 8, r.coat);
  rect(ctx, x - 4, y - 6, 4, 6, r.light);
  rect(ctx, x - 8, y - 5, 3, 6, P.ink);
  rect(ctx, x + 6, y - 5, 3, 6, P.ink);
  rect(ctx, x - 7, y - 4, 2, 4, r.skin);
  rect(ctx, x + 6, y - 4, 2, 4, r.skin);
  rect(ctx, x - 5, y - 17, 11, 10, P.ink);
  rect(ctx, x - 4, y - 16, 9, 8, r.skin);
  rect(ctx, x - 5, y - 18, 11, 5, r.hair);
  rect(ctx, x - 4, y - 18, 9, 2, index === 1 ? r.light : r.coat);
  rect(ctx, x - 5, y - 14, 2, 4, r.hair);
  if (index === 2) rect(ctx, x - 7, y - 16, 15, 2, r.light);
  if (index === 3) rect(ctx, x + 5, y - 16, 3, 7, r.hair);
  rect(ctx, x - 2, y - 12, 1, 2, P.ink);
  rect(ctx, x + 2, y - 12, 1, 2, P.ink);
  rect(ctx, x, y - 8, 2, 1, r.light);

  rect(ctx, x - 7, y + 13, 15, 13, P.ink);
  rect(ctx, x - 6, y + 14, 13, 11, selected ? r.coat : P.indigo);
  ctx.fillStyle = selected ? P.ink : P.cream;
  ctx.font = "bold 10px monospace";
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  ctx.fillText("ABCD"[index], x + 1, y + 20);
  if (limited) {
    // Current opportunity marker; never an assertion of permanent exclusion.
    rect(ctx, x + 10, y - 14, 10, 11, P.ink);
    ctx.fillStyle = P.coral;
    ctx.font = "bold 8px monospace";
    ctx.fillText("<1", x + 15, y - 8);
  }
}

function packet(ctx, start, end, progress, color) {
  // Six positions keep the animation discrete and anchored on the pixel grid.
  const t = 0.18 + Math.round(progress * 5) / 5 * 0.66;
  const x = Math.round(start.x + (end.x - start.x) * t);
  const y = Math.round(start.y + (end.y - start.y) * t);
  const direction = Math.atan2(end.y - start.y, end.x - start.x);
  // Two trailing dots encode direction without implying a transport mechanism.
  for (let i = 1; i <= 2; i++) {
    rect(ctx, x - Math.round(Math.cos(direction) * (5 + i * 4)),
      y - Math.round(Math.sin(direction) * (5 + i * 4)), 2, 2, color);
  }
  rect(ctx, x - 4, y - 4, 9, 9, P.ink);
  rect(ctx, x - 3, y - 3, 7, 7, P.indigo);
  flower(ctx, x, y, color, false);
}

/** Paint exact selected round's presentation; never advances numerical state. */
export function drawTown(canvas, {
  round, phase = 0, selectedResident = null, reducedMotion = false, progress = 0,
} = {}) {
  if (canvas.width !== TOWN_WIDTH) canvas.width = TOWN_WIDTH;
  if (canvas.height !== TOWN_HEIGHT) canvas.height = TOWN_HEIGHT;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  if (!backdrop) backdrop = buildBackdrop();
  ctx.drawImage(backdrop, 0, 0);
  const nextMissing = phase === 2 && (round?.pool_after == null);
  const pool = phase === 2 ? round?.pool_after : round?.pool_before;
  commons(ctx, pool, phase, nextMissing);
  residentPositions.forEach((_, i) => resident(ctx, i, selectedResident === i, (round?.offers?.[i] ?? 1) < 1));
  if (!round) return;

  const t = reducedMotion ? 0.5 : Math.min(1, Math.max(0, progress));
  residentPositions.forEach((position, i) => {
    const station = {x: position.x + (i % 2 ? -10 : 10), y: position.y - 3};
    const source = {x: i % 2 ? 282 : 198, y: i < 2 ? 121 : 145};
    if (phase === 0 && round.offers[i] > 0) packet(ctx, source, station, t, P.cyan);
    if (phase === 1) {
      if (round.contributions[i] > 0) packet(ctx, station, source, t, P.mint);
      if (round.surplus[i] > 0) {
        packet(ctx, {x: position.x, y: position.y - 4},
          {x: position.homeX, y: position.homeY - 1}, t, P.amber);
      }
    }
  });
}

/** Return a resident index for Canvas mouse/pointer events, or null. */
export function hitResident(canvas, event) {
  const bounds = canvas.getBoundingClientRect();
  const x = (event.clientX - bounds.left) * TOWN_WIDTH / bounds.width;
  const y = (event.clientY - bounds.top) * TOWN_HEIGHT / bounds.height;
  const index = residentPositions.findIndex(p => Math.abs(x - p.x) <= 20 && y >= p.y - 24 && y <= p.y + 30);
  return index >= 0 ? index : null;
}
