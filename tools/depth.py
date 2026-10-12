import sys, time, numpy as np, cv2, onnxruntime as ort
model = sys.argv[1]
img = cv2.cvtColor(cv2.imread('./work/counter.png'), cv2.COLOR_BGR2RGB)
H0, W0 = img.shape[:2]
s = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
name = s.get_inputs()[0].name
def run(rgb):
    x = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
    return s.run(None, {name: x})[0].reshape(518, 518)
t = time.time()
# 1) global pass (aspect squashed, then stretched back)
TH = 1036; TW = int(round(W0 * TH / H0))
big = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_CUBIC)
g = cv2.resize(run(img), (TW, TH), interpolation=cv2.INTER_CUBIC)
# 2) overlapping tiles at 2x density, each aligned to the global pass by least squares (scale + shift)
T, step = 518, 300
acc = np.zeros((TH, TW), np.float64); wsum = np.zeros((TH, TW), np.float64)
win1 = np.hanning(T + 2)[1:-1]; win = np.outer(win1, win1) + 1e-3
ys = list(range(0, TH - T + 1, step)); xs = list(range(0, TW - T + 1, step))
if ys[-1] != TH - T: ys.append(TH - T)
if xs[-1] != TW - T: xs.append(TW - T)
for y in ys:
    for x in xs:
        d = run(big[y:y+T, x:x+T]); ref = g[y:y+T, x:x+T]
        A = np.stack([d.ravel(), np.ones(d.size)], 1); a, b = np.linalg.lstsq(A, ref.ravel(), rcond=None)[0]
        acc[y:y+T, x:x+T] += (a * d + b) * win; wsum[y:y+T, x:x+T] += win
fine = acc / wsum
print('tiles', len(ys) * len(xs), 'time', round(time.time() - t, 1), 's')
out = cv2.resize(fine.astype(np.float32), (W0, H0), interpolation=cv2.INTER_AREA)
np.save('./work/disp.npy', out)
v = (out - out.min()) / (out.max() - out.min())
cv2.imwrite('./work/disp.png', (v * 255).astype(np.uint8))
