# EvoPolis · Final Fantasy V-inspired visual direction

EvoPolis uses an original visual language inspired by **Final Fantasy V**: lush layered landscapes, expressive small resident sprites, detailed tile-built villages, and silver-edged royal-blue menus. The user's reference pairs green countryside and distant blue mountains with a four-person party display. The working observatory translates that composition into a town and resident roster, retaining the deliberate pixels and readable menus of the SNES era.

The illustrated cover is a title-screen interpretation of this direction. The live town remains a deterministic Canvas scene with its own original art; it does not use the cover as a substitute for working visualization.

This direction applies to the cover, diagrams, empirical figures, and the working local community viewer. It changes how the research is presented, while keeping the experiment's definitions and evidence explicit.

## The world on screen

- A top-down or three-quarter **2D town**, with clustered foliage, layered blue-green mountains, limestone paths, timber cottages, shaded roof tiles, a shared commons, and a community hall.
- **Four distinct resident sprites** for the initial four-person protocol. Additional decorative characters should not look like participating agents.
- Contributions and allocations appear as short, discrete resource-transfer animations. Movement depicts an action; it does not silently introduce distance, transport costs, or spatial rules into the experiment.
- A **party-style resident roster** and status panels show the current round, available pool, allocations, returns, and private surplus using actual recorded or simulated values. Fantasy styling does not introduce health, magic, combat, jobs, or character traits into the experiment.
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

Interface windows add a separate royal-blue ramp (`#263F83` → `#142C65` → `#10214F`) and silver trim (`#A8B7D1`). These surface colors do not replace the stable research-series encoding above. Town foliage and masonry use their own clustered sprite palettes.

Mechanism plots use a stable mapping: **Equal = coral/circle; Mixed = amber/square; Proportional = cyan/triangle; Recorded RL = mint/diamond**. Label every condition. Resident identity colors in the town are a separate encoding and must not imply that a resident represents one of these institutions.

## Geometry, type, and motion

| Element | Direction |
| :--- | :--- |
| Tiles and sprites | Start from a deliberate pixel grid; use integer scaling and nearest-neighbor rendering. |
| Windows | Square or stepped corners, crisp two-tone borders, restrained hard shadows. |
| Titles | An ivory fantasy title in cover art and the masthead; monospaced labels in interface panels. |
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

Stages 02–04 implement pixel-art replay for empirical records, scripted simulations, trained-agent communities and forecasts from observed human history, including saved pre-choice predictions. The forecast view keeps the observed prefix, generated branch, actual human continuation and post-termination padding visibly distinct. Its pointwise forecast bands retain the numerical scales and mark the precise observed/forecast boundary; they are not confidence intervals over human groups. The [real forecast screenshot](assets/forecast-community-viewer.png) shows the working comparison and resident inspection. Further work should preserve visible resource transfers, accurate status panels, resident inspection, pause/step/play controls, and synchronized comparisons on common numerical scales. Keep recorded trajectories, fixed scripts and fitted-model predictions visibly distinct.

Future interfaces should make the experiment and model behavior observable through this visual language. They should not change the social mechanism in order to make the scene resemble a game.
