"""Classical baselines for the running example (Fashion-MNIST).

k-NN and a random forest on raw pixels in [0, 1], with the MLP's protocol:
55,000 training images, 5,000 held-out validation images (used only to choose
k), and the 10,000 official test images, scored once.

Data: the four .gz files from github.com/zalandoresearch/fashion-mnist
(data/fashion). Usage:
    python classical_baselines.py knn [data_dir]
    python classical_baselines.py rf SEED [data_dir]      # SEED = 0, 1, 2
"""
import gzip
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def load(data_dir, kind):
    with gzip.open(f"{data_dir}/{kind}-labels-idx1-ubyte.gz") as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    with gzip.open(f"{data_dir}/{kind}-images-idx3-ubyte.gz") as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(len(y), 784)
    return x.astype(np.float32) / 255.0, y


def splits(data_dir):
    x_all, y_all = load(data_dir, "train")
    x_te, y_te = load(data_dir, "t10k")
    perm = np.random.default_rng(0).permutation(len(y_all))
    tr, va = perm[:55_000], perm[55_000:]
    return x_all[tr], y_all[tr], x_all[va], y_all[va], x_te, y_te


def knn_votes(idx, y_tr, k):
    """Majority vote over the k nearest; ties go to the lower class index, as in scikit-learn."""
    lab = y_tr[idx[:, :k]]
    counts = np.zeros((len(lab), 10), int)
    for j in range(k):
        counts[np.arange(len(lab)), lab[:, j]] += 1
    return counts.argmax(1)


def run_knn(data_dir):
    x_tr, y_tr, x_va, y_va, x_te, y_te = splits(data_dir)
    nn = KNeighborsClassifier(n_neighbors=9, algorithm="brute").fit(x_tr, y_tr)
    idx_va = nn.kneighbors(x_va, return_distance=False)
    accs = {k: (knn_votes(idx_va, y_tr, k) == y_va).mean() for k in (1, 3, 5, 7, 9)}
    for k, a in accs.items():
        print(f"k-NN k={k}: validation {a:.4f}")
    k = max(accs, key=accs.get)
    t = time.time()
    idx_te = nn.kneighbors(x_te, return_distance=False)
    print(f"k-NN k={k}: TEST {(knn_votes(idx_te, y_tr, k) == y_te).mean():.4f} ({time.time() - t:.0f} s)")


def run_rf(seed, data_dir):
    x_tr, y_tr, x_va, y_va, x_te, y_te = splits(data_dir)
    rf = RandomForestClassifier(100, n_jobs=-1, random_state=seed).fit(x_tr, y_tr)
    print(f"RF seed={seed}: validation {rf.score(x_va, y_va):.4f}  TEST {rf.score(x_te, y_te):.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "knn":
        run_knn(*sys.argv[2:])
    else:
        run_rf(int(sys.argv[2]), *sys.argv[3:])
