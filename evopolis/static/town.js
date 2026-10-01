/* Original EvoPolis pixel art, composed for a 480 × 240 logical canvas.
 * The layered landscape, clustered foliage and character silhouettes draw on
 * the visual language of 16-bit JRPGs. No game artwork or sprites are reused.
 * Scenery is presentation only: distance, homes and movement add no game rules.
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
  ink: '#182c38', indigo: '#243d6c', slate: '#48648a', cream: '#fff2ce',
  pale: '#bbcfe0', cyan: '#6dcff6', mint: '#73e0b3', amber: '#ffcf6e', coral: '#ff8b8b',
  grass: '#73a14c', grassDark: '#59803d', grassLight: '#9dbb5e', meadow: '#86ad51',
  treeDark: '#1d493c', treeShade: '#2a603e', tree: '#397b42', leaf: '#579546',
  leafLight: '#7dab4f', leafSun: '#a3c364', soil: '#655247',
  path: '#c3b895', pathDark: '#8a8e73', pathLight: '#e1d4ac',
  wall: '#e3d9af', stone: '#c2c5a2', stoneShade: '#949f89',
  wood: '#705343', woodLight: '#b48b58', woodShade: '#4b433d',
};

// Personal colors are independent of the institution colors in research plots.
const residents = [
  {coat: '#8273b7', light: '#cec5ed', roof: '#74759d', roofLight: '#a4a1bf', roofDark: '#515975', hair: '#493d66', skin: '#eec59d', skinShade: '#c48f70'},
  {coat: '#65939b', light: '#d4e7df', roof: '#548d8c', roofLight: '#87b4a0', roofDark: '#3a676f', hair: '#7b513d', skin: '#c69773', skinShade: '#986b57'},
  {coat: '#c98155', light: '#f0c77d', roof: '#a36754', roofLight: '#d59a69', roofDark: '#78524e', hair: '#643e3b', skin: '#deb48c', skinShade: '#b18367'},
  {coat: '#a59b62', light: '#f4e4b6', roof: '#8a8964', roofLight: '#b9b38a', roofDark: '#646b56', hair: '#635075', skin: '#bf9073', skinShade: '#956d59'},
];

let backdrop;

function rect(ctx, x, y, width, height, color) {
  ctx.fillStyle = color;
  ctx.fillRect(Math.round(x), Math.round(y), width, height);
}

// Vertices are integral; stepped contours keep scenery on its native pixel grid.
function shape(ctx, points, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  ctx.closePath();
  ctx.fill();
}

function flower(ctx, x, y, color, stem = true) {
  if (stem) {
    rect(ctx, x, y + 2, 1, 5, P.treeShade);
    rect(ctx, x + 1, y + 4, 2, 1, P.leafLight);
  }
  rect(ctx, x - 1, y - 2, 3, 5, color);
  rect(ctx, x - 2, y - 1, 5, 3, color);
  rect(ctx, x, y, 1, 1, P.cream);
}

function leafCluster(ctx, x, y, size, light = false) {
  const w = size, h = Math.floor(size * .7);
  const dark = light ? P.tree : P.treeShade;
  const mid = light ? P.leafLight : P.leaf;
  const sun = light ? P.leafSun : P.leafLight;
  // Overlapping, stepped lobes give a tree a canopy rather than a tiled outline.
  rect(ctx, x + 3, y, w - 7, h, dark);
  rect(ctx, x, y + 4, w, h - 8, dark);
  rect(ctx, x + 1, y + 2, w - 3, h - 3, dark);
  rect(ctx, x + 3, y + 1, w - 7, h - 4, mid);
  rect(ctx, x + 1, y + 4, w - 6, h - 8, mid);
  rect(ctx, x + 5, y, w - 11, 2, sun);
  rect(ctx, x + 2, y + 3, 4, 2, sun);
  rect(ctx, x + 7, y + 4, 4, 2, sun);
  rect(ctx, x + w - 6, y + h - 5, 4, 3, dark);
  rect(ctx, x + 4, y + h - 4, 4, 2, P.treeShade);
  rect(ctx, x + 8, y + 7, 2, 2, dark);
}

function tree(ctx, x, y, variant = 0) {
  const shift = variant % 3 - 1;
  rect(ctx, x - 15, y + 24, 32, 3, P.grassDark);
  rect(ctx, x - 11, y + 27, 25, 2, P.grassDark);
  rect(ctx, x - 3, y + 10, 7, 16, P.treeDark);
  rect(ctx, x - 2, y + 10, 4, 15, P.wood);
  rect(ctx, x - 2, y + 14, 1, 8, P.woodLight);
  rect(ctx, x - 5, y + 23, 11, 3, P.treeDark);
  rect(ctx, x - 8, y + 10, 7, 3, P.woodShade);
  rect(ctx, x + 1, y + 8, 8, 3, P.woodShade);
  leafCluster(ctx, x - 21, y - 1, 24, false);
  leafCluster(ctx, x - 4, y - 4, 25, false);
  leafCluster(ctx, x - 15, y + 7, 27, false);
  leafCluster(ctx, x - 14 + shift, y - 14, 27, false);
  leafCluster(ctx, x - 18, y - 5, 22, true);
  leafCluster(ctx, x - 6 + shift, y - 12, 22, true);
  leafCluster(ctx, x + 2, y + 1, 18, false);
  rect(ctx, x - 15, y + 11, 3, 2, P.leafLight);
  rect(ctx, x + 8, y + 8, 3, 1, P.leafLight);
  rect(ctx, x - 8, y - 8, 2, 1, P.cream);
}

function shrub(ctx, x, y, blossoms = false) {
  leafCluster(ctx, x, y, 15, false);
  leafCluster(ctx, x + 8, y + 2, 12, true);
  if (blossoms) {
    flower(ctx, x + 5, y + 4, '#ecc6aa', false);
    flower(ctx, x + 13, y + 6, '#c0b8db', false);
  }
}

function fence(ctx, x, y, length) {
  rect(ctx, x, y + 2, length, 3, P.woodShade);
  rect(ctx, x, y + 2, length, 1, P.woodLight);
  rect(ctx, x, y + 7, length, 2, P.wood);
  for (let i = 0; i <= length; i += 10) {
    rect(ctx, x + i, y, 3, 12, P.woodShade);
    rect(ctx, x + i, y, 2, 10, P.woodLight);
    rect(ctx, x + i, y, 2, 2, P.pathLight);
  }
}

function barrel(ctx, x, y) {
  rect(ctx, x + 1, y + 1, 9, 11, P.woodShade);
  rect(ctx, x, y + 3, 11, 7, P.woodShade);
  rect(ctx, x + 2, y + 2, 7, 9, P.woodLight);
  rect(ctx, x + 3, y + 2, 1, 9, P.wood);
  rect(ctx, x + 7, y + 2, 1, 9, P.wood);
  rect(ctx, x + 1, y + 3, 9, 1, P.stoneShade);
  rect(ctx, x + 1, y + 8, 9, 1, P.stoneShade);
  rect(ctx, x + 2, y, 7, 2, P.pathDark);
  rect(ctx, x + 3, y, 5, 1, P.pathLight);
}

function windowPane(ctx, x, y) {
  rect(ctx, x, y, 11, 12, P.woodShade);
  rect(ctx, x + 1, y + 1, 9, 9, '#527c8f');
  rect(ctx, x + 2, y + 2, 3, 3, '#c8e1d3');
  rect(ctx, x + 6, y + 2, 2, 3, '#97bdb7');
  rect(ctx, x + 5, y + 1, 1, 9, P.woodLight);
  rect(ctx, x + 1, y + 5, 9, 1, P.woodLight);
  rect(ctx, x - 2, y + 1, 2, 10, P.wood);
  rect(ctx, x + 11, y + 1, 2, 10, P.wood);
  rect(ctx, x - 2, y + 11, 15, 2, P.wall);
  rect(ctx, x - 1, y + 13, 13, 1, P.stoneShade);
}

function tiledRoof(ctx, x, y, width, r) {
  const rows = 5;
  for (let row = 0; row < rows; row++) {
    const inset = (rows - row - 1) * 4;
    rect(ctx, x + inset - 1, y + row * 4, width - inset * 2 + 2, 5, P.woodShade);
    rect(ctx, x + inset, y + row * 4, width - inset * 2, 3, r.roof);
    rect(ctx, x + inset, y + row * 4, width - inset * 2, 1, r.roofLight);
    for (let tx = x + inset + (row % 2 ? 3 : 7); tx < x + width - inset - 2; tx += 8) {
      rect(ctx, tx, y + row * 4 + 1, 1, 3, r.roofDark);
      rect(ctx, tx + 1, y + row * 4 + 1, 3, 1, r.roofLight);
    }
  }
  rect(ctx, x - 2, y + 20, width + 4, 2, P.woodShade);
  rect(ctx, x - 2, y + 20, width + 3, 1, r.roofDark);
  rect(ctx, x + 16, y - 1, width - 32, 2, r.roofLight);
}

function house(ctx, x, y, index, garden) {
  const r = residents[index];
  rect(ctx, x + 2, y + 47, 63, 6, P.grassDark);
  rect(ctx, x + 5, y + 50, 60, 4, P.grassDark);
  rect(ctx, x + 1, y + 14, 58, 36, P.woodShade);
  rect(ctx, x + 3, y + 17, 52, 29, P.wall);
  rect(ctx, x + 54, y + 18, 4, 29, P.stoneShade);
  rect(ctx, x + 3, y + 43, 51, 5, P.stone);
  // Exposed frame, limestone foundations and occasional masonry joints.
  rect(ctx, x + 3, y + 17, 3, 28, P.wood);
  rect(ctx, x + 51, y + 17, 3, 28, P.wood);
  rect(ctx, x + 5, y + 18, 46, 2, P.woodLight);
  rect(ctx, x + 3, y + 40, 51, 2, P.wood);
  for (let i = 0; i < 5; i++) {
    rect(ctx, x + 8 + i * 10, y + 44, 1, 3, P.stoneShade);
    rect(ctx, x + 6 + i * 10, y + 43, 8, 1, P.cream);
  }
  // Chimney sits behind the roof, with individual worn stones.
  rect(ctx, x + 43, y - 10, 9, 23, P.woodShade);
  rect(ctx, x + 44, y - 8, 6, 20, P.stone);
  for (let cy = y - 7; cy < y + 9; cy += 4) {
    rect(ctx, x + 44, cy, 6, 1, P.stoneShade);
    rect(ctx, x + 46 + (cy % 3), cy + 1, 1, 2, P.stoneShade);
  }
  rect(ctx, x + 42, y - 11, 11, 3, P.stoneShade);
  rect(ctx, x + 43, y - 11, 9, 1, P.pathLight);
  tiledRoof(ctx, x - 3, y - 4, 66, r);
  windowPane(ctx, x + 10, y + 24);
  windowPane(ctx, x + 39, y + 24);
  rect(ctx, x + 26, y + 25, 11, 22, P.woodShade);
  rect(ctx, x + 28, y + 27, 7, 19, P.wood);
  rect(ctx, x + 29, y + 28, 1, 16, P.woodLight);
  rect(ctx, x + 32, y + 28, 1, 16, P.woodShade);
  rect(ctx, x + 33, y + 36, 1, 2, P.amber);
  rect(ctx, x + 25, y + 46, 14, 2, P.pathLight);
  rect(ctx, x + 23, y + 48, 18, 2, P.stoneShade);
  rect(ctx, x + 23, y + 48, 17, 1, P.stone);
  if (garden) {
    rect(ctx, x + 8, y + 38, 15, 4, P.woodShade);
    rect(ctx, x + 9, y + 39, 13, 2, P.woodLight);
    flower(ctx, x + 11, y + 37, '#f4dca1', false);
    flower(ctx, x + 18, y + 36, '#d5afd2', false);
  } else {
    barrel(ctx, x + (index === 1 ? -8 : 55), y + 38);
  }
  // A resident-colored pennant identifies the home without an extra actor.
  rect(ctx, x + 55, y + 20, 8, 1, P.woodShade);
  rect(ctx, x + 60, y + 21, 5, 8, r.roofDark);
  rect(ctx, x + 60, y + 21, 4, 6, r.light);
  rect(ctx, x + 61, y + 23, 2, 3, r.coat);
}

function hall(ctx) {
  const x = 210, y = 27;
  rect(ctx, x - 3, y + 40, 70, 5, P.grassDark);
  rect(ctx, x + 1, y + 8, 60, 34, P.woodShade);
  rect(ctx, x + 4, y + 12, 54, 28, P.stone);
  rect(ctx, x + 7, y + 13, 48, 23, P.wall);
  for (let row = 0; row < 4; row++) {
    rect(ctx, x + 5, y + 17 + row * 6, 52, 1, P.stoneShade);
    for (let col = 0; col < 4; col++) {
      rect(ctx, x + 10 + col * 12 + row % 2 * 5, y + 13 + row * 6, 1, 4, P.stoneShade);
    }
  }
  tiledRoof(ctx, x - 4, y - 10, 70, {
    roof: '#53758b', roofLight: '#90afb4', roofDark: '#3d5c75',
  });
  // A small civic bell turret; unlike a battle UI, no invented combat symbols.
  rect(ctx, x + 23, y - 21, 15, 17, P.woodShade);
  rect(ctx, x + 25, y - 19, 11, 13, P.wall);
  rect(ctx, x + 28, y - 17, 5, 8, P.woodShade);
  rect(ctx, x + 29, y - 15, 3, 4, P.woodLight);
  rect(ctx, x + 28, y - 11, 5, 1, P.amber);
  rect(ctx, x + 21, y - 23, 19, 3, P.slate);
  rect(ctx, x + 24, y - 25, 13, 2, '#90afb4');
  rect(ctx, x + 7, y + 13, 4, 26, P.pathLight);
  rect(ctx, x + 50, y + 13, 4, 26, P.pathLight);
  windowPane(ctx, x + 13, y + 15);
  windowPane(ctx, x + 39, y + 15);
  rect(ctx, x + 27, y + 19, 11, 20, P.woodShade);
  rect(ctx, x + 28, y + 20, 9, 19, P.wood);
  rect(ctx, x + 29, y + 21, 1, 17, P.woodLight);
  rect(ctx, x + 34, y + 28, 1, 2, P.amber);
  rect(ctx, x + 21, y + 39, 23, 2, P.pathLight);
  rect(ctx, x + 18, y + 41, 29, 2, P.stoneShade);
  rect(ctx, x + 16, y + 43, 33, 2, P.pathLight);
}

function bench(ctx, x, y) {
  rect(ctx, x + 2, y + 6, 2, 6, P.woodShade);
  rect(ctx, x + 17, y + 6, 2, 6, P.woodShade);
  rect(ctx, x, y, 22, 3, P.woodShade);
  rect(ctx, x + 1, y, 20, 2, P.woodLight);
  rect(ctx, x, y + 4, 22, 4, P.woodShade);
  rect(ctx, x + 1, y + 4, 20, 2, P.woodLight);
  rect(ctx, x + 2, y + 4, 18, 1, P.pathLight);
}

function lantern(ctx, x, y) {
  rect(ctx, x + 1, y + 2, 2, 15, P.woodShade);
  rect(ctx, x + 1, y + 3, 1, 11, P.woodLight);
  rect(ctx, x - 3, y, 9, 2, P.ink);
  rect(ctx, x - 2, y - 7, 7, 7, P.ink);
  rect(ctx, x - 1, y - 6, 5, 5, P.woodLight);
  rect(ctx, x, y - 5, 2, 4, P.cream);
  rect(ctx, x - 1, y - 9, 5, 2, P.ink);
  rect(ctx, x - 1, y + 15, 6, 2, P.treeDark);
}

function horizon(ctx) {
  rect(ctx, 0, 0, 480, 40, '#99bac4');
  rect(ctx, 0, 0, 480, 7, '#729aa9');
  rect(ctx, 0, 7, 480, 8, '#87aeba');
  // Distant mountain ranges are a landscape backdrop, never game geography.
  shape(ctx, [[0, 30], [0, 16], [18, 16], [18, 12], [28, 12], [28, 7], [39, 7], [39, 3], [48, 3], [48, 9], [65, 9], [65, 16], [93, 16], [93, 10], [103, 10], [103, 4], [118, 4], [118, 12], [130, 12], [130, 19], [153, 19], [153, 12], [174, 12], [174, 7], [184, 7], [184, 14], [201, 14], [201, 30]], '#658b9e');
  shape(ctx, [[276, 30], [276, 17], [292, 17], [292, 11], [305, 11], [305, 4], [319, 4], [319, 11], [331, 11], [331, 18], [349, 18], [349, 12], [364, 12], [364, 5], [376, 5], [376, 12], [392, 12], [392, 18], [412, 18], [412, 11], [427, 11], [427, 3], [438, 3], [438, 11], [451, 11], [451, 16], [465, 16], [465, 10], [480, 10], [480, 30]], '#648da0');
  [[39, 3], [103, 4], [305, 4], [364, 5], [427, 3]].forEach(([x, y]) => {
    rect(ctx, x, y, 9, 2, '#d8e6d7');
    rect(ctx, x - 3, y + 2, 7, 2, '#bdcfcc');
    rect(ctx, x + 2, y + 4, 4, 2, '#91b7c1');
  });
  for (let x = -5; x < 480; x += 18) {
    const rise = (x * 7 + 97) % 9;
    rect(ctx, x, 24 + rise, 24, 11, '#517d6c');
    rect(ctx, x + 5, 21 + rise, 12, 16, '#517d6c');
    rect(ctx, x + 8, 24 + rise, 4, 2, '#729a72');
  }
  rect(ctx, 0, 35, 480, 3, '#62844e');
}

function buildBackdrop() {
  const canvas = document.createElement('canvas');
  canvas.width = TOWN_WIDTH;
  canvas.height = TOWN_HEIGHT;
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  rect(ctx, 0, 0, 480, 240, P.grass);
  horizon(ctx);
  // Broad meadow patches and tiny grass clusters use fixed integer coordinates.
  // Seeking or comparing communities therefore cannot change the scenery.
  [[86, 42, 101, 32], [288, 44, 114, 30], [110, 119, 71, 42], [309, 117, 66, 45],
    [113, 203, 78, 29], [290, 202, 80, 29], [189, 72, 107, 16]].forEach(([x, y, w, h]) => {
    rect(ctx, x + 4, y, w - 8, h, P.meadow);
    rect(ctx, x, y + 5, w, h - 10, P.meadow);
  });
  for (let y = 39; y < 240; y += 5) {
    for (let x = 3; x < 480; x += 7) {
      const seed = (x * 73 + y * 29) % 31;
      if (seed < 7) {
        rect(ctx, x + seed % 3, y, 1, 2, P.grassDark);
        rect(ctx, x + seed % 3 + 2, y + 1, 1, 1, P.grassDark);
      } else if (seed > 25) {
        rect(ctx, x, y, 2, 1, P.grassLight);
        if (seed === 30) rect(ctx, x + 2, y - 1, 1, 1, P.grassLight);
      }
    }
  }

  // The roads preserve the original four stations and shared crossroads.
  const paths = [[66, 82, 350, 16], [66, 177, 350, 16], [228, 63, 24, 145],
    [68, 68, 17, 24], [395, 68, 17, 24], [68, 163, 17, 23], [395, 163, 17, 23],
    [185, 93, 109, 85]];
  paths.forEach(([x, y, w, h]) => {
    rect(ctx, x - 3, y - 2, w + 6, h + 4, P.grassDark);
    rect(ctx, x - 1, y - 1, w + 2, h + 2, P.pathDark);
    rect(ctx, x, y, w, h, P.path);
    rect(ctx, x, y, w, 1, P.pathLight);
  });
  for (let y = 65; y < 206; y += 6) {
    for (let x = 65 + y % 4; x < 418; x += 9) {
      if (paths.some(([px, py, w, h]) => x >= px + 1 && x + 7 < px + w && y >= py + 1 && y + 4 < py + h)) {
        const worn = (x * 3 + y) % 7;
        rect(ctx, x, y, 7, 4, worn < 2 ? '#b3ae8e' : worn > 4 ? '#cec39e' : P.path);
        rect(ctx, x, y, 6, 1, P.pathLight);
        rect(ctx, x + 7, y + 1, 1, 4, P.pathDark);
        rect(ctx, x + 1, y + 4, 6, 1, P.pathDark);
      }
    }
  }

  // Dense woodland frames the community; no background figures imply agents.
  [[8, 26], [33, 30], [119, 26], [153, 25], [184, 30], [292, 29], [324, 25], [363, 27], [448, 28], [475, 29],
    [5, 64], [22, 65], [461, 67], [481, 60], [4, 106], [22, 116], [461, 107], [485, 108],
    [3, 155], [23, 164], [462, 156], [486, 158], [4, 201], [24, 217], [461, 207], [485, 214]]
    .forEach(([x, y], i) => tree(ctx, x, y, i));
  fence(ctx, 44, 110, 61);
  fence(ctx, 374, 110, 61);
  fence(ctx, 44, 207, 61);
  fence(ctx, 374, 207, 61);
  [[148, 41], [303, 42], [145, 213], [302, 214]].forEach(([x, y], i) => {
    rect(ctx, x - 1, y - 1, 24, 12, P.woodShade);
    rect(ctx, x, y, 22, 10, P.soil);
    for (let j = 0; j < 4; j++) flower(ctx, x + 3 + j * 5, y + 2 + j % 2 * 3, residents[i].light);
    rect(ctx, x, y + 10, 22, 2, P.woodLight);
  });
  [[111, 48], [108, 139], [354, 47], [355, 140], [157, 125], [305, 125],
    [58, 117], [397, 118], [185, 211], [277, 211]].forEach(([x, y], i) => shrub(ctx, x, y, i % 3 === 0));
  house(ctx, 45, 28, 0, true);
  house(ctx, 373, 28, 1, false);
  house(ctx, 45, 123, 2, false);
  house(ctx, 373, 123, 3, true);
  hall(ctx);
  bench(ctx, 154, 67);
  bench(ctx, 305, 67);
  bench(ctx, 199, 204);
  bench(ctx, 261, 204);
  lantern(ctx, 175, 164);
  lantern(ctx, 306, 164);
  // A stone well is scenery, separate from the numerical commons.
  rect(ctx, 147, 154, 23, 8, P.grassDark);
  rect(ctx, 149, 151, 19, 11, P.stoneShade);
  rect(ctx, 151, 153, 15, 7, P.stone);
  rect(ctx, 155, 153, 1, 7, P.stoneShade);
  rect(ctx, 161, 153, 1, 7, P.stoneShade);
  rect(ctx, 151, 156, 15, 1, P.stoneShade);
  rect(ctx, 148, 150, 21, 3, P.pathLight);
  rect(ctx, 152, 150, 13, 2, P.treeDark);
  rect(ctx, 154, 151, 9, 1, '#53878b');
  rect(ctx, 149, 138, 2, 14, P.wood);
  rect(ctx, 166, 138, 2, 14, P.wood);
  rect(ctx, 146, 136, 25, 3, P.woodShade);
  rect(ctx, 149, 133, 19, 3, P.woodLight);
  rect(ctx, 152, 131, 13, 2, P.wood);
  rect(ctx, 158, 139, 1, 10, P.pathLight);
  barrel(ctx, 320, 148);
  barrel(ctx, 333, 154);

  // Foreground tree crowns add depth without covering stations or their labels.
  [[11, 245], [48, 248], [86, 253], [121, 252], [162, 256], [320, 255], [361, 253], [399, 253], [439, 248], [478, 242]]
    .forEach(([x, y], i) => tree(ctx, x, y, i + 2));
  [[179, 230], [291, 231], [207, 225], [259, 226]].forEach(([x, y], i) => {
    flower(ctx, x, y, i % 2 ? '#d6c1dc' : '#eee0a1');
    rect(ctx, x + 4, y + 4, 2, 1, P.grassDark);
  });
  return canvas;
}

function commons(ctx, pool, phase, missingAfter) {
  // Raised limestone garden; its blooms remain a schematic stock indicator.
  rect(ctx, 196, 155, 90, 7, P.pathDark);
  rect(ctx, 195, 111, 90, 43, P.woodShade);
  rect(ctx, 198, 108, 84, 48, P.stoneShade);
  rect(ctx, 198, 109, 84, 5, P.pathLight);
  rect(ctx, 198, 112, 4, 42, P.wall);
  rect(ctx, 279, 112, 3, 42, P.stoneShade);
  rect(ctx, 202, 114, 76, 38, P.treeDark);
  rect(ctx, 204, 117, 72, 32, P.soil);
  rect(ctx, 200, 151, 80, 6, P.stone);
  rect(ctx, 200, 151, 80, 2, P.pathLight);
  for (let x = 204; x < 280; x += 10) {
    rect(ctx, x, 153, 1, 4, P.stoneShade);
    rect(ctx, x + 1, 156, 8, 1, P.stoneShade);
  }
  [[196, 110], [279, 110], [196, 151], [279, 151]].forEach(([x, y]) => {
    rect(ctx, x, y, 6, 7, P.stoneShade);
    rect(ctx, x, y, 6, 2, P.cream);
    rect(ctx, x, y + 2, 2, 4, P.wall);
  });
  // Positive fractional pools retain a bloom; their exact values stay in the UI.
  const flowerCount = typeof pool === 'number' && Number.isFinite(pool)
    ? Math.min(28, Math.max(0, Math.ceil(pool / 200 * 28))) : 0;
  for (let i = 0; i < 28; i++) {
    const x = 210 + (i % 7) * 9;
    const y = 121 + Math.floor(i / 7) * 7;
    rect(ctx, x - 2, y + 3, 5, 1, P.woodShade);
    if (i < flowerCount) {
      rect(ctx, x - 2, y + 3, 2, 1, P.leafLight);
      flower(ctx, x, y, i % 4 === 0 ? '#f5e4ae' : P.mint);
    }
  }
  rect(ctx, 208, 98, 64, 14, P.woodShade);
  rect(ctx, 209, 98, 62, 1, P.pathLight);
  rect(ctx, 209, 99, 62, 11, P.indigo);
  rect(ctx, 210, 100, 60, 1, P.slate);
  rect(ctx, 211, 103, 1, 3, P.woodLight);
  rect(ctx, 268, 103, 1, 3, P.woodLight);
  ctx.fillStyle = P.cream;
  ctx.font = 'bold 8px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('COMMONS', 240, 105);
  if (phase === 2) {
    const color = missingAfter ? P.pale : P.mint;
    [[193, 106, 7, 2], [193, 106, 2, 7], [280, 106, 7, 2], [285, 106, 2, 7],
      [193, 160, 7, 2], [193, 155, 2, 7], [280, 160, 7, 2], [285, 155, 2, 7]]
      .forEach(([x, y, w, h]) => rect(ctx, x, y, w, h, color));
    if (missingAfter) {
      rect(ctx, 232, 126, 16, 16, P.ink);
      ctx.fillStyle = P.cream;
      ctx.font = 'bold 12px monospace';
      ctx.fillText('?', 240, 134);
    }
  }
}

// Hand-drawn sprite maps: hair, face, clothing folds, hands and separate boots.
// They are original civilian silhouettes; outfits encode identity, not abilities.
const SPRITES = [
  [
    '      ooooo       ', '    oohhhhhoo     ', '   ohhHHHhhhho    ',
    '   ohHhhhHHhho    ', '  ohhHhhhhhHhho   ', '  ohhhhhhhhhho    ',
    '  ohhsSSSSShho    ', '  ohsSSSSSSsho    ', '  ohsSeSSeSsho    ',
    '   hsSSSSSshh     ', '   ohssSSshho     ', '    ohssshho      ',
    '   oooLggLooo     ', '  occlLggLlcCo    ', ' occclLLLlccCco   ',
    ' occclLglcccCco   ', ' ocsclccclccsCo   ', ' ossclccclccsso   ',
    '  osclccclccso    ', '   oclLLLlcco     ', '   oclccclcco     ',
    '   occlllccco     ', '   ooccccco oo    ', '    opoooppo      ',
    '    oppooppo      ', '    obboobbo      ', '   obbboobbbo     ',
  ],
  [
    '      ooooo       ', '     ohhhhhho     ', '    ohHHHhhhho    ',
    '   ohHHhhhhhhho   ', '   ohhhhHhhhhho   ', '   ohhSSSShhho    ',
    '   ohSSSSSShho    ', '   osSSSSSSsho    ', '   osSeSSeSsho    ',
    '    sSSSSSSso     ', '    ossSSsso      ', '    oossssoo      ',
    '   occggggcco     ', '  ocCLLLLcccco    ', ' ocCLcLLcLcccco   ',
    ' ocCLcLLcLcccco   ', ' osCLcLLcLccCso   ', ' ossLcLLcLccsso   ',
    '  osLggggLccso    ', '   ocLLcLLcco     ', '   oocLcLccoo     ',
    '    opoooppo      ', '    oppooppo      ', '    oppooppo      ',
    '    obboobbo      ', '    obboobbo      ', '   obbboobbbo     ',
  ],
  [
    '      ooooo       ', '    oohhhhhoo     ', '   ohHHHhhhhho    ',
    '   ohHHhhhhhhho   ', '   oLLLLLLLLLoo   ', '  oLLggggggLLLLo  ',
    '   ohSSSSSShho    ', '   osSSSSSSsho    ', '   osSeSSeSsho    ',
    '    sSSSSSSso     ', '    ossSSsso      ', '    oossssoo      ',
    '   ocLLccLLco     ', '  ocCLLccLLcco    ', ' ocCCLLccLLccco   ',
    ' ocCCLcccLLccco   ', ' osCCLcccLLccso   ', ' ossCLcccLLcsso   ',
    '  osCLgggLLcso    ', '   ocLLcLLcco     ', '   oocLLLccoo     ',
    '    opoooppo      ', '    oppooppo      ', '    oppooppo      ',
    '    obboobbo      ', '    obboobbo      ', '   obbboobbbo     ',
  ],
  [
    '      ooooo       ', '     oLLLLLo      ', '    oLggLLLLo     ',
    '   oLLLhhLLLLo    ', '   oLLhhhhLLLo    ', '   oLhSSSShLLo    ',
    '   ohSSSSSShho    ', '   osSSSSSShho    ', '   osSeSSeShho    ',
    '    sSSSSSShho    ', '    ossSSshhho    ', '    oossshhho     ',
    '   oLgggggLLho    ', '  oLLLcLcLLLho    ', ' oCLLLcLcLLLLco   ',
    ' oCLLLcLcLLLLco   ', ' osCLLcLcLLLCso   ', ' ossLLcLcLLLsso   ',
    '  osLLgggLLLso    ', '   oLLcLcLLLo     ', '   oLLcLcLLLo     ',
    '   oLLcLcLLLo     ', '   ooLLLLLLoo     ', '    opoooppo      ',
    '    obboobbo      ', '    obboobbo      ', '   obbboobbbo     ',
  ],
];

function drawSprite(ctx, index, x, y) {
  const r = residents[index];
  const colors = {
    o: P.ink, h: r.hair, H: index === 1 ? '#ad7c50' : '#8d7190',
    S: r.skin, s: r.skinShade, e: P.ink, c: r.coat, C: r.roofDark,
    L: r.light, l: r.roofLight, g: P.woodLight, p: P.slate, b: P.woodShade,
  };
  SPRITES[index].forEach((row, sy) => {
    [...row].forEach((pixel, sx) => {
      if (colors[pixel]) rect(ctx, x + sx, y + sy, 1, 1, colors[pixel]);
    });
  });
}

/** The roster and town share the same original character artwork. */
export function drawResidentPortrait(canvas, index) {
  if (canvas.width !== 24) canvas.width = 24;
  if (canvas.height !== 32) canvas.height = 32;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 24, 32);
  ctx.imageSmoothingEnabled = false;
  if (!Number.isInteger(index) || index < 0 || index >= residents.length) return;
  drawSprite(ctx, index, 3, 2);
}

function resident(ctx, index, selected, limited) {
  const {x, y} = residentPositions[index];
  const r = residents[index];
  rect(ctx, x - 7, y + 6, 15, 3, P.pathDark);
  rect(ctx, x - 10, y + 7, 21, 1, P.pathDark);
  if (selected) {
    [[x - 11, y - 22, 7, 2], [x - 11, y - 22, 2, 7], [x + 5, y - 22, 7, 2], [x + 10, y - 22, 2, 7],
      [x - 11, y + 10, 7, 2], [x - 11, y + 5, 2, 7], [x + 5, y + 10, 7, 2], [x + 10, y + 5, 2, 7]]
      .forEach(([rx, ry, w, h]) => rect(ctx, rx, ry, w, h, P.cream));
  }
  // Every resident stays visible when their available offer is below one.
  drawSprite(ctx, index, x - 8, y - 20);
  rect(ctx, x - 7, y + 13, 15, 13, P.ink);
  rect(ctx, x - 6, y + 14, 13, 11, selected ? r.coat : P.indigo);
  rect(ctx, x - 6, y + 14, 13, 1, selected ? r.light : P.slate);
  ctx.fillStyle = P.cream;
  ctx.font = 'bold 10px monospace';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  ctx.fillText('ABCD'[index], x + 1, y + 20);
  if (limited) {
    // This is current opportunity, never an assertion of permanent exclusion.
    rect(ctx, x + 10, y - 14, 12, 12, P.ink);
    rect(ctx, x + 11, y - 13, 10, 10, P.indigo);
    ctx.fillStyle = P.coral;
    ctx.font = 'bold 8px monospace';
    ctx.fillText('<1', x + 16, y - 8);
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
