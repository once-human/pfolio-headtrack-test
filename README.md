# pfolio-headtrack-test

A feasibility test for my portfolio: the screen behaves like a pane of glass above a real-looking kitchen counter. Move your head (or the laptop) and the scene shifts in true 3D.

Live: https://pfolio-headtrack-test.vercel.app

## How it works

1. **One photograph.** `assets/photo.jpg` is an AI-generated overhead shot of the counter.
2. **Depth, offline.** Depth Anything V2 (Large) estimates depth on a global pass plus 18 overlapping high-res tiles, each aligned to the global pass. That keeps edges sharp enough to separate single mint leaves.
3. **Heights.** A smooth surface is fitted to the green tiles and subtracted, so the tiles read as flat and everything on them gets a height (scaled so the tumbler is about as tall as it is wide).
4. **Layers.**
   - `assets/table.jpg`: the tiles with the objects removed. Underneath each object the fill darkens like a contact shadow, and past the photo's edges it falls off into darkness.
   - `assets/height.png`: per-pixel height (R = high byte, G = low byte, in 1/16 px) and the object mask (B). Very tall spikes are softened (a standing bottle seen slightly from the side reads as a huge spike to a monocular depth model), and soft depth ramps at object edges are snapped into clean steps.
   - `assets/behind.jpg`: on the high side of every depth step, the colour of the surface just beyond it.
5. **Render.** WebGL2. The tiles are one textured quad. The objects are a textured depth mesh: a vertex every 2 photo pixels (~250k), slid along the camera's ray so the resting view reproduces the photo exactly. Colour and edge alpha are sampled per fragment straight from the photo and mask, so it is as sharp as the source. Where you lean and an edge opens up, the stretched cliff surface (detected per pixel from the height gradient over the photo) is painted with the darkened colour of what's behind, so the gap reads as the object's shadow instead of a smear. (The gaussian-splat renderer is still there for `.ply` / `.splat` captures.)
6. **Window effect.** Off-axis (asymmetric frustum) projection from the head position: left/right, up/down and tilt. Distance is locked, so there's no zoom.
7. **Head tracking.** MediaPipe Face Landmarker on every webcam frame. An adaptive One Euro filter smooths while you're still and opens up when you move, with a short damped prediction for camera latency and a ~40 ms critically damped spring.

**Privacy-safe preview:** the camera feed is never shown. A small corner card draws a lamp-lit plaster mask from the tracker's 3D face mesh, framed like the camera's view, plus your distance from the screen.

**Performance:** no live backdrop blur or animated blend-mode overlays (both cost a full-screen pass every frame), and the preview card redraws at most ~30 times a second.

**Depth follows distance:** lean toward the screen and the scene's depth grows; sit back and it flattens (1.5x range). The framing never zooms.

Renders at the screen's native resolution. Needs HTTPS (or localhost) for the camera.

Controls: Parallax, Depth (height exaggeration) and Tilt sliders, Recenter, and `H` to hide the UI. Drag in a `.ply` / `.splat` capture to view a real gaussian-splat scan instead.

## Swapping the photo

Put the new image at `work/counter.png`, then:

```
pip install onnxruntime opencv-contrib-python-headless numpy
# model: https://github.com/fabio-sim/Depth-Anything-ONNX/releases (depth_anything_v2_vitl.onnx)
python tools/depth.py depth_anything_v2_vitl.onnx      # -> work/disp.npy
python tools/prep.py                                    # -> work/photo.jpg, table_ext.jpg, height.png
```

Copy `photo.jpg`, `table_ext.jpg` (renamed `table.jpg`) and `height.png` into `assets/`.

Notes for a new photo:
- Shoot or generate it perfectly overhead, sharp everywhere (no shallow depth of field).
- Leave some empty margin at the edges.
- `prep.py` assumes green tiles when fitting the table surface; change the hue range for other surfaces.
- A higher-resolution source (e.g. 4K) gives a crisper result.
