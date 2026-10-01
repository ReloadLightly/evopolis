# EvoPolis · 16-bit visual direction

EvoPolis uses an original visual language inspired by NES and SNES games of the late 1980s and early 1990s: tile-based towns, expressive small sprites, limited palettes, square dialogue boxes, and clear status displays. The main reference is the richer 16-bit look of the SNES era, with the crisp simplicity of NES menus.

This direction applies to the cover, diagrams, empirical figures, and the working local community viewer. It changes how the research is presented, while keeping the experiment's definitions and evidence explicit.

## The world on screen

- A top-down or three-quarter **2D town**, built from readable tiles: paths, homes, trees, a shared commons, and a community hall.
- **Four distinct resident sprites** for the initial four-person protocol. Additional decorative characters should not look like participating agents.
- Contributions and allocations appear as short, discrete resource-transfer animations. Movement depicts an action; it does not silently introduce distance, transport costs, or spatial rules into the experiment.
- A **status panel** shows the current round, available pool, allocations, returns, and private surplus using actual recorded or simulated values.
- Selecting a resident opens a **dialogue-style information panel** with their observed history, available information, and model predictions when such predictions exist.
- Side-by-side communities use the same layout and clock so that differences between institutions are easy to see.

Use original art and characters. The identity comes from the era's visual grammar, rather than copied game assets.

## Palette

The reusable interface variables are in [evopolis-theme.css](assets/evopolis-theme.css).

| Role | Color | Use |
| :--- | :--- | :--- |
| Midnight | `#172038` | Main background and deep outlines. |
| Indigo | `#253A5E` | Panels and chart plotting areas. |
| Slate | `#3B5278` | Raised edges and muted grid lines. |
| Parchment | `#FFF1D2` | Primary text, important values, highlights. |
| Soft blue | `#B6C6DF` | Supporting text and annotations. |
| Mint | `#73E0B3` | Resource returns and model-search paths. |
| Cyan | `#6DCFF6` | Allocations, information, and comparison accents. |
| Amber | `#FFCF6E` | Retained resources and institutional-search paths. |
| Coral | `#FF8B8B` | Attention states and a contrasting series color. |

Mechanism plots use a stable mapping: **Equal = coral/circle; Mixed = amber/square; Proportional = cyan/triangle; Recorded RL = mint/diamond**. Label every condition. Resident identity colors in the town are a separate encoding and must not imply that a resident represents one of these institutions.

## Geometry, type, and motion

| Element | Direction |
| :--- | :--- |
| Tiles and sprites | Start from a deliberate pixel grid; use integer scaling and nearest-neighbor rendering. |
| Windows | Square or stepped corners, crisp two-tone borders, restrained hard shadows. |
| Titles | Short block/pixel lettering in artwork; bold monospaced labels in interface panels. |
| Prose | Readable system text with normal spacing and comfortable line height. |
| Values and axes | Clear monospaced numerals; preserve decimals, units, and minus signs. |
| Motion | Brief movement or resource-transfer cues linked to real actions; pause and step controls remain usable. |
| Focus and selection | An obvious high-contrast outline or cursor, with keyboard access. |

Use flat color clusters and purposeful dithering in artwork. Avoid photorealism, smooth 3D miniatures, glossy glass panels, blurred neon glows, and decorative scanlines over text. Pixel fonts belong in short labels, not dense methods paragraphs.

## Research graphics

Architecture diagrams resemble game information screens, with square-framed panels, pixel residents, clear arrows, and concise labels. The same palette connects them to the town.

Empirical plots use matching frames, typography, and series colors. **Data positions, uncertainty intervals, axes, and annotations remain accurately rendered.** Do not apply a pixelation filter to a plot, round values for visual effect, turn continuous values into decorative bars, or replace measured points with generated artwork. Preserve editable SVG exports and the code that produced them.

Provide distinct source labels for **recorded human data**, **recorded upstream model outcomes**, and **new EvoPolis simulation**. Concept art is labeled separately. The visual theme should help a reader identify which kind of evidence they are seeing.

## GitHub and the local interface

GitHub controls README prose, table, and code-block typography. Keep those sections as accessible, searchable Markdown; the illustrated cover, diagrams, and plotted figures carry the retro identity. Do not convert research tables into pictures just to imitate a game menu.

The local community viewer applies the supplied CSS tokens to its original Canvas town, controls, inspection panels and charts. See the [viewer guide](viewer.md) for the working application and a real screenshot. Scope styles beneath `.ep-root` when embedding the interface in another application.

## Continuing the interface

Stages 02–03 implement pixel-art replay for empirical records, scripted simulations and trained-agent communities, including saved pre-choice predictions. Further work should preserve visible resource transfers, accurate status panels, resident inspection, pause/step/play controls, and synchronized comparisons on common numerical scales. Keep recorded trajectories, fixed scripts and fitted-model predictions visibly distinct.

Future interfaces should make the experiment and model behavior observable through this visual language. They should not change the social mechanism in order to make the scene resemble a game.
