# pfolio-headtrack-test

A feasibility test for my portfolio: a head-tracked gaussian-splat scene where the screen behaves like a window.

- **Scene:** an evening dinner table (cocktail, cutting board, steaming pan, candle), built procedurally from gaussian splats.
- **Renderer:** custom WebGL2 splat renderer. The camera only translates, so splats are depth-sorted once at load and never re-sorted.
- **Window effect:** off-axis (asymmetric frustum) projection driven by head position. Left/right, up/down and head tilt (roll); distance is locked, so there's no zoom.
- **Head tracking:** MediaPipe Face Landmarker on the webcam, smoothed with a One Euro filter and a critically damped spring.
- **Render:** native screen resolution (CSS size × devicePixelRatio). Splat density is set from the screen's pixel width at load (~4 device pixels apart, about 1.1M splats at 1080p) so surfaces and edges stay crisp.
- **Responsiveness:** every camera frame is processed; an adaptive One Euro filter smooths only when you're still, a short velocity prediction covers camera/model latency, and a fast critically damped spring rounds off frame steps.
- **Captures:** drag in a `.ply` (3DGS trainer output) or `.splat` file to view a real capture.

Single static file: `index.html`. Needs HTTPS (or localhost) for the camera.

Keys: `H` hides the UI.
