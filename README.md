# pfolio-headtrack-test

A feasibility test for my portfolio: the screen behaves like a pane of glass above a real-looking kitchen counter. Move your head (or the laptop) and the scene shifts in true 3D.

Live: https://pfolio-headtrack-test.vercel.app

## How it works

1. **One photograph.** `assets/photo.jpg` is an AI-generated overhead shot of the counter.
2. **Depth, offline.** Depth Anything V2 (Large) estimates depth on a global pass plus 18 overlapping high-res tiles, each aligned to the global pass. That keeps edges sharp enough to separate single mint leaves.
3. **Heights.** A smooth surface is fitted to the green tiles and subtracted, so the tiles read as flat and everything on them gets a height (scaled so the tumbler is about as tall as it is wide).
4. **Layers.**
   - `assets/table.jpg`: the tiles with the objects removed. Underneath each object the fill darkens like a contact shadow, and past the photo's edges it falls off into darkness.
   - `assets/height.png`: per-pixel height (R = high byte, G = low byte, in 1/16 px) and the object mask (B).
5. **Render.** WebGL2. The tiles are one textured quad. Each object pixel (~960k) becomes a gaussian splat, placed along the ray the camera saw it on, so the resting view reproduces the photo exactly. Splats are stretched along the surface slope so steep sides stay closed. They are sorted once by depth; the camera only translates and rolls, so the order never changes.
6. **Window effect.** Off-axis (asymmetric frustum) projection from the head position: left/right, up/down and tilt. Distance is locked, so there's no zoom.
7. **Head tracking.** MediaPipe Face Landmarker on every webcam frame. An adaptive One Euro filter smooths while you're still and opens up when you move, with a short damped prediction for camera latency and a ~40 ms critically damped spring.

**Privacy-safe preview:** the camera feed is never shown. The corner card draws a lamp-lit plaster mask built only from the tracker's 3D face mesh (~900 triangles), framed like the camera's view, so you see what is tracked without any video.

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
