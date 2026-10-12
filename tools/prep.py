import numpy as np, cv2, json
img = cv2.imread('./work/counter.png')                 # BGR
H, W = img.shape[:2]
disp = np.load('./work/disp.npy').astype(np.float64)
yy, xx = np.mgrid[0:H, 0:W]
# 1) robust table surface: smooth cubic polynomial fitted to green-tile pixels, trimming outliers
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
green = (hsv[..., 0] > 38) & (hsv[..., 0] < 95) & (hsv[..., 1] > 70) & (hsv[..., 2] > 25)
u = xx.ravel() / W - 0.5; v = yy.ravel() / H - 0.5
A = np.stack([u**i * v**j for i in range(4) for j in range(4) if i + j <= 3], 1); d = disp.ravel()
sel = green.ravel().copy()
for _ in range(10):
    p = np.linalg.lstsq(A[sel], d[sel], rcond=None)[0]
    r = d - A @ p
    sel = green.ravel() & (r < np.percentile(r[green.ravel()], 60))
res = (d - A @ p).reshape(H, W)
res -= np.median(res[sel.reshape(H, W)])
# 2) scale to pixel units: tumbler rim (bottom-left) ~ 210 px tall
rim = np.percentile(res[600:830, 190:420], 99.5)
k = 210.0 / rim
h = res * k
tn = h[sel.reshape(H,W)]; print('rim res', rim, 'k', k, 'bottle cap px', np.percentile(h[200:280, 1450:1550], 99), 'board px', np.median(h[450:650, 600:900]), 'tile noise p50/p95', np.percentile(np.abs(tn),50), np.percentile(np.abs(tn),95))
# 3) snap depth edges to colour edges with a guided filter
guide = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
hg = cv2.ximgproc.guidedFilter(guide, h.astype(np.float32), 4, 25.0)
hg = np.maximum(hg, 0)
# 4) object mask: anything clearly above the tiles (threshold from the tile noise)
THR = max(9.0, float(np.percentile(np.abs(hg[sel.reshape(H,W)]), 97)) * 1.15); print('thr', THR)
m = (hg > THR).astype(np.uint8)
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
keep = np.zeros(n, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] > 60
m = keep[lab].astype(np.uint8)
md = cv2.dilate(m, np.ones((9, 9), np.uint8))                     # foreground reaches a few px past the hole, so the fill is never seen head-on
alpha = cv2.GaussianBlur(md.astype(np.float32), (0, 0), 1.2)
alpha = np.clip(alpha * 1.6, 0, 1)
print('object pixels', int((alpha > 0.02).sum()), 'of', H * W)
# 5) table under the objects: streak-free push-pull fill, darkening with depth under the object
#    (what you glimpse under an object's edge is in its shadow in real life, so this reads naturally)
hole = cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(np.uint8)
valid = (1 - hole).astype(np.float32)
def pushpull(im, w):
    pyr = [(im * w[..., None], w)]
    while min(pyr[-1][1].shape) > 8:
        a_, b_ = pyr[-1]; pyr.append((cv2.pyrDown(a_), cv2.pyrDown(b_)))
    acc, wacc = pyr[-1]
    est = acc / np.maximum(wacc, 1e-6)[..., None]
    for a_, b_ in reversed(pyr[:-1]):
        up = cv2.resize(est, (a_.shape[1], a_.shape[0]), interpolation=cv2.INTER_LINEAR)
        here = a_ / np.maximum(b_, 1e-6)[..., None]
        k = np.clip(b_ * 4, 0, 1)[..., None]
        est = here * k + up * (1 - k)
    return est
fill = pushpull(img.astype(np.float32), valid)
dist = cv2.distanceTransform(hole, cv2.DIST_L2, 5)
occl = (0.66 + 0.34 * np.exp(-dist / 30.0))[..., None]
noise = cv2.GaussianBlur(np.random.default_rng(1).normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.9)[..., None] * 3.5
fill = fill * occl + noise
hm = hole.astype(np.float32)[..., None]
bg = np.clip(img * (1 - hm) + fill * hm, 0, 255).astype(np.uint8)
# 6) edges: the lamp's light falls off toward the borders, and past the photo the counter
#    becomes a soft, dim blur, so leaning far shows darkness instead of a seam or mirrored tiles
PAD = 600
bd = np.minimum(np.minimum(xx, W - 1 - xx), np.minimum(yy, H - 1 - yy)).astype(np.float32)
M = 0.07 * W
t_ = np.clip(bd / M, 0, 1); edgefade = (0.6 + 0.4 * t_ * t_ * (3 - 2 * t_))[..., None]
bg = np.clip(bg * edgefade, 0, 255).astype(np.uint8)
img_v = np.clip(img * edgefade, 0, 255).astype(np.uint8)          # same falloff on the photo the splats use
ext = cv2.copyMakeBorder(bg, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101).astype(np.float32)
ey, ex = np.mgrid[0:H + 2*PAD, 0:W + 2*PAD]
dx = np.maximum(0, np.maximum(PAD - ex, ex - (W + PAD - 1))) / PAD
dy = np.maximum(0, np.maximum(PAD - ey, ey - (H + PAD - 1))) / PAD
r = np.sqrt(dx**2 + dy**2)
soft = cv2.GaussianBlur(ext, (0, 0), 22)
wb = np.clip(r * 14, 0, 1)[..., None]
ext = ext * (1 - wb) + soft * wb
fall = (0.12 + 0.88 * np.exp(-r * 7.5))[..., None]
ext = np.clip(ext * fall, 0, 255).astype(np.uint8)
cv2.imwrite('./work/table_ext.jpg', ext, [cv2.IMWRITE_JPEG_QUALITY, 93])
cv2.imwrite('./work/photo.jpg', img_v, [cv2.IMWRITE_JPEG_QUALITY, 96])
# 7) height (16-bit, 1/16 px units) + alpha packed into one PNG
hq = np.clip(hg * 16, 0, 65535).astype(np.uint16)
pack = np.dstack([(alpha * 255).astype(np.uint8), (hq & 255).astype(np.uint8), (hq >> 8).astype(np.uint8)])   # BGR order -> R=hi, G=lo, B=alpha
cv2.imwrite('./work/height.png', pack)
json.dump({'w': W, 'h': H, 'pad': PAD, 'heightScale': 1 / 16}, open('./work/meta.json', 'w'))
# previews
vis = img.copy(); vis[alpha > 0.5] = (vis[alpha > 0.5] * 0.4 + np.array([255, 0, 255]) * 0.6).astype(np.uint8)
cv2.imwrite('./work/prev_mask.jpg', cv2.resize(vis, (W // 2, H // 2)))
cv2.imwrite('./work/prev_bg.jpg', cv2.resize(bg, (W // 2, H // 2)))
