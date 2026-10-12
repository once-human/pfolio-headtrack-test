# pfolio-headtrack-test

A feasibility test for my portfolio: a head-tracked gaussian-splat scene where the screen behaves like a window.

- **Scene:** an evening dinner table (cocktail, cutting board, steaming pan, candle), built procedurally from ~600k gaussian splats.
- **Renderer:** custom WebGL2 splat renderer. The camera only translates, so splats are depth-sorted once at load and never re-sorted.
- **Window effect:** off-axis (asymmetric frustum) projection driven by head position. Left/right and up/down; distance is locked, so there's no zoom.
- **Head tracking:** MediaPipe Face Landmarker on the webcam, smoothed with a One Euro filter and a critically damped spring.
- **Render:** 4K by default (1080p option). Rendering above the window size gives supersampled edges.
- **Captures:** drag in a `.ply` (3DGS trainer output) or `.splat` file to view a real capture.

Single static file: `index.html`. Needs HTTPS (or localhost) for the camera.

Keys: `H` hides the UI.
