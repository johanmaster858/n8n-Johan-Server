# Breathing Room

A 40-second film (1920×1080, 30 fps). A stone cathedral's single pointed-arch window is slowly bricked up by cubes engraved with *Busy*, *Efficiency*, *Urgent* and *Deadline*, until the room is dark and silent. Then one cube lets go. Light pours back in, coloured by a stained-glass window of the Virgin Mary with two angels and a rose, until its image covers the floor. The film ends on the JAGA TECH emblem igniting in the dark.

No people appear in the film. Everything is procedural: the geometry, the stained glass, the light and the sound.

## Output

| file | what |
| --- | --- |
| `output/breathing_room.mp4` | the film with its procedural soundtrack (H.264 High, BT.709, CRF 20 + AAC 256 kb/s, 1200 frames) |
| `output/stained_glass_design.png` | the original, symmetrical window design every light effect is computed from |
| `output/jaga_tech_emblem_traced.svg` | the emblem vectorised from `assets/logo final 111.jpeg` (99.3 % IoU with the reference) |
| `output/contact_sheet.jpg` | one frame per second |

## Timeline

| time | scene |
| --- | --- |
| 0–3 s | Colourless white light. A soft, window-shaped patch fades up on the floor. |
| 3–11 s | Gray cubes fly in from the foreground and fill the window, faster and faster. The floor patch breaks up and shrinks. |
| 11–14 s | The window is sealed: darkness and silence. As the eye adapts, the engraved words glow faintly. |
| 14–19 s | The *BUSY* cube over Mary's praying hands trembles and slides out. It falls and dissolves into particles of light. A red, blue and gold shaft falls through the gap. |
| 19–26 s | Chain reaction outward from that opening. Every gap adds a coloured column of light and uncovers more of the glass. |
| 25.4–33 s | The camera glides into the light columns, cranes up through them and turns to look straight down. The Virgin and the angels cover the floor in coloured light, upright by 32.6 s. |
| 33–36 s | The light swells until the whole frame is white. *Abundance lives in the space between.* |
| 36–40 s | The light implodes into shadow. The emblem rises as a dark silhouette against a cold halo, then its letters J·A·G·A / TECH ignite in electric-blue neon and its circuitry in gold, and they pulse. Cut to black at 39.8 s. |

## How it is made

* `stained_glass.py` draws the window as a map of glass pieces, mirrors it for exact symmetry, then leads, colours and paints it. The result is a transmission texture.
* `scene.py` is a numba ray caster covering the nave, pillars, the window recess and the cubes, including their dissolve.
  * Every lit point is projected back along the 45° sun direction onto the glass plane and samples that same texture.
  * That lookup is multiplied by a per-cell *cover* map (cubes sitting in the window) and by a depth map of moving cubes.
  * So the window as seen from inside, the coloured light columns (single-scattering ray-march) and the floor projection always match, and blocked cells cast shadows.
* `timeline.py` holds every animated quantity as a pure function of `t`: cube flights, the chain-reaction release times, the camera path, light levels and eye adaptation. Each frame is rendered at exactly `t = frame / 30` and stands on its own. Fast cubes get extra temporal sub-samples for motion blur.
* `post.py` does bloom, ACES tone mapping and the colour grade (cold and desaturated first, then jewel red, blue and gold). It also draws the titles in Shippori Mincho and adds film grain.
* `logo.py` builds the emblem sequence:
  * It traces the reference logo into sub-pixel vector contours and renders them with a polar-space volumetric ray pass, which gives the stark shadows.
  * It splits the emblem into letters, circuitry and face lines so that the letters and circuitry ignite.
* `audio.py` synthesises the soundtrack on the same timeline: clock, whooshes and thuds, the slam and the silence, chimes, a warm pad, then the ignition.

## Rendering

```bash
pip install numpy scipy pillow opencv-python-headless numba scikit-image imageio-ffmpeg fonttools
python render.py assets                 # stained glass + textures -> build/
python render.py still 22.0 test.png    # any single moment
python render.py frames 0 1200          # PNG sequence -> build/frames/f0000.png ...
python audio.py build/soundtrack.wav
python render.py encode                 # -> output/breathing_room.mp4 (+ a silent copy)
```

`render_all.sh` runs the whole pipeline with two worker processes.

Each frame takes about 2–6 s on 4 CPU cores at 1080p. Set `BR_W=960 BR_H=540` for a fast preview.

Fonts: Shippori Mincho (Latin subset) and Oswald, both under the SIL Open Font License.
