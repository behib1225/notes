"""fmnist_regularization.py -- regularization vs. augmentation on 1,000 Fashion-MNIST images.

Pure NumPy (no framework), so every step is visible. Same MLP as the running example:
784-128-64-10, ReLU, He-normal init, zero biases, cross-entropy on raw logits.

Configs (3 seeds each, seeds 0, 1, 2):
  1k  none          : Adam, lr 1e-3
  1k  dropout + wd  : dropout p = 0.5 after each hidden ReLU, AdamW (decoupled) wd 0.5 on weights only
  1k  augmentation  : random resized crop (80-100 % area, aspect 3/4-4/3) + horizontal flip p = 0.5
  1k  flip only     : horizontal flip p = 0.5 only (no crop)
  1k  all three     : augmentation (crop + flip) + dropout + wd
  55k none          : the full training split, 20 epochs, batch 128 (as the running example)
1k runs: batch 100, 150 epochs. Inputs standardized with (x - 0.286) / 0.353.
Data split (fixed, seed 1234): 55,000 train / 5,000 validation from the 60,000 training images;
the 1,000-image subset is the first 1,000 of the shuffled 55,000. The 10,000 test images are used once
per run, at the checkpoint with the lowest validation loss.

Output: results.json (per-epoch validation curves and per-run summaries).
Data: fashion-mnist *.gz files (github.com/zalandoresearch/fashion-mnist) in ./fm/
"""
import gzip, json, time
import numpy as np
from PIL import Image

MEAN, STD = 0.286, 0.353


def load(prefix):
    with gzip.open(f"fm/{prefix}-images-idx3-ubyte.gz") as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(f"fm/{prefix}-labels-idx1-ubyte.gz") as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return x.astype(np.float32) / 255.0, y


def standardize(x):  # x: (m, 28, 28) in [0, 1]
    return ((x - MEAN) / STD).reshape(len(x), -1).astype(np.float32)


def augment(x, rng, crop=True):
    """Random resized crop (area 0.8-1.0, aspect 3/4-4/3) then horizontal flip with p = 0.5."""
    out = np.empty_like(x)
    for i, img in enumerate(x):
        if not crop:
            out[i] = img[:, ::-1] if rng.random() < 0.5 else img
            continue
        for _ in range(10):
            area = rng.uniform(0.8, 1.0) * 784
            ar = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
            w, h = int(round(np.sqrt(area * ar))), int(round(np.sqrt(area / ar)))
            if 0 < w <= 28 and 0 < h <= 28:
                break
        else:
            w = h = 28
        top, left = rng.integers(0, 28 - h + 1), rng.integers(0, 28 - w + 1)
        crop = Image.fromarray(np.ascontiguousarray(img[top:top + h, left:left + w]))  # float32 'F' image
        res = np.asarray(crop.resize((28, 28), Image.BILINEAR), dtype=np.float32)
        out[i] = res[:, ::-1] if rng.random() < 0.5 else res
    return out


class MLP:
    sizes = [784, 128, 64, 10]

    def __init__(self, rng):
        self.W = [rng.normal(0, np.sqrt(2 / a), (b, a)).astype(np.float32)
                  for a, b in zip(self.sizes[:-1], self.sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in self.sizes[1:]]

    def params(self):
        return self.W + self.b

    def forward(self, X, p_drop=0.0, rng=None):
        cache, a = [], X
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W.T + b
            if l < len(self.W) - 1:
                h = np.maximum(z, 0)
                mask = None
                if p_drop > 0:
                    mask = (rng.random(h.shape) >= p_drop).astype(np.float32) / (1 - p_drop)
                    h = h * mask
                cache.append((a, z, mask))
                a = h
            else:
                cache.append((a, z, None))
                a = z
        return a, cache  # a = logits

    def backward(self, dlogits, cache):
        gW, gb, d = [None] * 3, [None] * 3, dlogits
        for l in reversed(range(3)):
            a_in, z, mask = cache[l]
            if l < 2:
                if mask is not None:
                    d = d * mask
                d = d * (z > 0)
            gW[l], gb[l] = d.T @ a_in, d.sum(0)
            if l > 0:
                d = d @ self.W[l]
        return gW + gb


def xent(logits, y):
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return -logp[np.arange(len(y)), y].mean(), np.exp(logp)


def evaluate(model, X, y, bs=1000):
    loss, correct = 0.0, 0
    for i in range(0, len(X), bs):
        logits, _ = model.forward(X[i:i + bs])
        l, _ = xent(logits, y[i:i + bs])
        loss += l * len(logits)
        correct += (logits.argmax(1) == y[i:i + bs]).sum()
    return loss / len(X), correct / len(X)


def run(Xtr_raw, ytr, Xval, yval, Xte, yte, seed, epochs, bs, aug=False, crop=True, p_drop=0.0, wd=0.0, lr=1e-3):
    rng = np.random.default_rng(seed)
    model = MLP(rng)
    P = model.params()
    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    Xtr_fixed = None if aug else standardize(Xtr_raw)
    hist, best = [], (np.inf, None, -1)
    for ep in range(epochs):
        Xtr = standardize(augment(Xtr_raw, rng, crop)) if aug else Xtr_fixed
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            logits, cache = model.forward(Xtr[idx], p_drop, rng)
            _, prob = xent(logits, ytr[idx])
            dlog = prob.copy()
            dlog[np.arange(len(idx)), ytr[idx]] -= 1
            grads = model.backward(dlog / len(idx), cache)
            t += 1
            for k, (p, g) in enumerate(zip(P, grads)):
                if wd > 0 and k < 3:          # decoupled decay, weights only (AdamW)
                    p -= lr * wd * p
                m[k] = b1 * m[k] + (1 - b1) * g
                v[k] = b2 * v[k] + (1 - b2) * g * g
                mh, vh = m[k] / (1 - b1 ** t), v[k] / (1 - b2 ** t)
                p -= lr * mh / (np.sqrt(vh) + eps)
        vl, va = evaluate(model, Xval, yval)
        hist.append((float(vl), float(va)))
        if vl < best[0]:
            best = (vl, [p.copy() for p in P], ep + 1)
    for p, q in zip(P, best[1]):
        p[...] = q
    tl, ta = evaluate(model, Xte, yte)
    return dict(val_curve=hist, best_epoch=best[2], best_val_loss=float(best[0]),
                val_acc_at_best=hist[best[2] - 1][1], test_loss=float(tl), test_acc=float(ta))


if __name__ == "__main__":
    X, y = load("train")
    Xte_raw, yte = load("t10k")
    perm = np.random.default_rng(1234).permutation(60000)
    tr, va = perm[:55000], perm[55000:]
    Xval, yval = standardize(X[va]), y[va]
    Xte = standardize(Xte_raw)
    sub = tr[:1000]
    configs = {
        "1k, none": dict(idx=sub, epochs=150, bs=100),
        "1k, dropout + wd": dict(idx=sub, epochs=150, bs=100, p_drop=0.5, wd=0.5),
        "1k, augmentation": dict(idx=sub, epochs=150, bs=100, aug=True),
        "1k, flip only": dict(idx=sub, epochs=150, bs=100, aug=True, crop=False),
        "1k, all three": dict(idx=sub, epochs=150, bs=100, aug=True, p_drop=0.5, wd=0.5),
        "55k, none": dict(idx=tr, epochs=20, bs=128),
    }
    results = {}
    for name, c in configs.items():
        c = dict(c)
        idx = c.pop("idx")
        results[name] = []
        for seed in range(3):
            t0 = time.time()
            r = run(X[idx], y[idx], Xval, yval, Xte, yte, seed, **c)
            results[name].append(r)
            print(f"{name:18s} seed {seed}: best ep {r['best_epoch']:3d}  best val loss {r['best_val_loss']:.3f}"
                  f"  val acc {r['val_acc_at_best']:.4f}  test acc {r['test_acc']:.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
    json.dump(results, open("results.json", "w"))
