# Autumn Echoes

A 40-second vertical short (1080 × 1920, 30 fps) rendered with three.js. It moves from the weight of possessions to the lightness of lived moments.

| Time | Scene |
| --- | --- |
| 0–6 s | A safe, a stack of books, a briefcase and a padlock rest heavily on the leaves of a Canadian maple forest. The light is muted and the air is still. |
| 6–14 s | A gust sweeps through. Each object burns away along a wind-driven front into amber embers and red maple leaves, and the image blooms from muted to vivid. The camera follows the wind up into the low sun. |
| 14–22 s | Memories: a hand holds a red maple leaf against the sun; worn boots crunch through dry leaves; a campfire crackles by a glassy lake at sunset. |
| 22–32 s | The camera lifts above the treeline over rolling hills of crimson, orange and gold, split by a winding blue river. |
| 32–36 s | The landscape melts into a glowing amber vignette. |
| 36–40 s | The JAGA TECH emblem, from `assets/logo final 111.jpeg`, kindles like an ember, breathes, cools and fades to black. |

On-screen words (Spanish):

| Time | Text |
| --- | --- |
| 0–5 s | *A veces, nos llenamos de cosas...* |
| 6–11 s | *...buscando una felicidad que no se toca.* |
| 14–18 s | *Mi felicidad no se mide en bienes,* |
| 19–23 s | *se mide en recuerdos y anhelos.* |
| 26–31 s | *Al final, solo somos las historias que vivimos.* |

## How it is made

Everything is procedural; no downloaded models, textures or samples are used. The only external inputs are the logo image and the Cormorant Garamond font.

- **Rendering:** three.js r170 runs in headless Chromium (SwiftShader WebGL2), driven by Playwright. The scene is rendered at `t = frame / 30` exactly, one frame at a time, and captured losslessly (`driver.py`, `render.py`).
- **Post chain** (`web/src/core/post.js`), all in HDR:
  - volumetric sunlight, ray-marched through the sun's shadow map with drifting noise;
  - analytic aerial perspective;
  - screen-space god rays;
  - scatter-as-gather depth of field with a foreground bleed;
  - dual-filter bloom;
  - ACES tone mapping and a colour grade.
- **Materials:** leaves come from a painted atlas of eight sugar-maple leaves with vein and translucency maps, lit with a backlit transmission term; canopies use leaf-cluster cards. The painted steel, leather, paper, bark, denim, wool and soil are canvas-generated PBR maps.
- **Organic models:** the hand and the boots are signed-distance fields polygonised with surface nets (`web/src/core/sdf.js`). The skin shader adds wrap lighting and subsurface glow.
- **Dissolve:** a world-space field combines the wind direction with 3D noise; the same noise is sampled in GLSL and JS. Each ember and leaf leaves the surface exactly when the burning front reaches its point.
- **Physics:** wind is mean flow plus gusts, curl-noise turbulence and a vortex. It drives embers and tumbling leaves, which have anisotropic drag, flutter torque and ground contact. The simulation steps at a fixed 1/120 s from t = 0, so every frame is reproducible.
- **Scene-specific shaders:**
  - the campfire flame is a ray-marched emissive volume;
  - the lake is a planar reflection;
  - the aerial forest is a canopy shader of per-tree crowns (species palette, stands, conifers), with LOD to stand colour;
  - the river is a sky-reflecting water shader.
- **Finishing** (`finish.py`):
  - light-bloom cross-dissolves between scenes;
  - the amber melt;
  - the ember emblem (`logo.py`), with kindling front, charcoal cracks and sparks;
  - Spanish titles with a soft glow and an adaptive shadow for legibility;
  - film grain and the H.264 encode.
- **Soundtrack** (`audio.py`) is synthesised from scratch and timed to the picture:
  - wind, leaves, footsteps, fire crackle, lake and a loon;
  - a Karplus–Strong nylon guitar, pads, strings and an ember bell.

## Reproduce

```bash
cd autumn-echoes/web && npm install              # three.js
cd .. && python3 render.py forest                 # and hand, boots, campfire, aerial
python3 audio.py build/soundtrack.wav
python3 finish.py build/video.mp4                 # picture
ffmpeg -i build/video.mp4 -i build/soundtrack.wav -c:v copy -c:a aac -b:a 192k -shortest output/autumn_echoes.mp4
```

Test stills: `python3 still.py SCENE t1 t2 ... [--w 540 --h 960]`.
