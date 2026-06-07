"""
benchmark/benchmark_streaming.py
=================================
Performance comparison between:
  1. Single DecisionTreeClassifier (streaming)
  2. RandomForestClassifier (streaming, n=10 trees)
  3. Loop-based vs NumPy-vectorised Gini computation

Outputs timing tables and a comparison plot.
Run: python benchmark/benchmark_streaming.py
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numcompute_stream.tree import DecisionTreeClassifier, _gini
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.io import generate_synthetic_dataset


# -----------------------------------------------------------------------
# 1. Benchmark: Single Tree vs Random Forest streaming accuracy
# -----------------------------------------------------------------------

def benchmark_model_comparison(n_samples=2000, n_features=10, n_chunks=20, n_classes=3):
    print("\n" + "=" * 60)
    print("BENCHMARK: Decision Tree vs Random Forest (Streaming)")
    print("=" * 60)
    print(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
    print(f"Chunks:  {n_chunks} × {n_samples // n_chunks} samples\n")

    X, y = generate_synthetic_dataset(n_samples, n_features, n_classes, random_state=42)
    y = y.astype(int)

    results = {}

    for name, model in [
        ("DecisionTree (depth=5)", DecisionTreeClassifier(max_depth=5, random_state=0)),
        ("RandomForest (n=10)", RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)),
    ]:
        trainer = StreamTrainer(model)
        t0 = time.perf_counter()
        trainer.run_stream(X, y, n_chunks=n_chunks, shuffle=True, random_state=0)
        elapsed = time.perf_counter() - t0

        acc_history = trainer.get_metric_history("accuracy")
        cum_acc = trainer.chunk_logs_[-1]["cumulative_accuracy"]

        results[name] = {
            "accuracy_history": acc_history,
            "cumulative_accuracy": cum_acc,
            "total_time": elapsed,
        }
        print(f"  {name:<35} CumAcc={cum_acc:.4f}  Time={elapsed:.3f}s")

    return results


# -----------------------------------------------------------------------
# 2. Benchmark: Loop vs Vectorised Gini impurity
# -----------------------------------------------------------------------

def benchmark_gini_vectorised_vs_loop(n_samples=10000, n_bins=50):
    print("\n" + "=" * 60)
    print("BENCHMARK: Loop vs Vectorised Split Scoring")
    print("=" * 60)
    print(f"Samples: {n_samples}, Threshold candidates: {n_bins}\n")

    rng = np.random.default_rng(0)
    X_col = rng.standard_normal(n_samples)
    y = (rng.standard_normal(n_samples) > 0).astype(int)
    thresholds = np.linspace(X_col.min(), X_col.max(), n_bins)

    # Loop version
    def gini_loop(y):
        if len(y) == 0:
            return 0.0
        n = len(y)
        gain = 0.0
        for label in np.unique(y):
            p = (y == label).sum() / n
            gain += p * p
        return 1.0 - gain

    t0 = time.perf_counter()
    for t in thresholds:
        left = y[X_col <= t]
        right = y[X_col > t]
        _ = gini_loop(left) + gini_loop(right)
    loop_time = time.perf_counter() - t0

    # Vectorised version (NumPy)
    def gini_vec(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return float(1.0 - np.dot(p, p))

    t0 = time.perf_counter()
    for t in thresholds:
        left = y[X_col <= t]
        right = y[X_col > t]
        _ = gini_vec(left) + gini_vec(right)
    vec_time = time.perf_counter() - t0

    speedup = loop_time / max(vec_time, 1e-9)
    print(f"  Loop version:        {loop_time * 1000:.2f} ms")
    print(f"  Vectorised version:  {vec_time * 1000:.2f} ms")
    print(f"  Speedup:             {speedup:.2f}×")

    return loop_time, vec_time, speedup


# -----------------------------------------------------------------------
# 3. Plot results
# -----------------------------------------------------------------------

def plot_benchmark_results(model_results, loop_time, vec_time, out_dir="benchmark"):
    os.makedirs(out_dir, exist_ok=True)

    # --- Accuracy over chunks ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("NumCompute Streaming Benchmark", fontsize=14, fontweight="bold")

    colors = ["#2563EB", "#16A34A"]
    for ax, (name, res), color in zip(
        [axes[0], axes[0]],
        model_results.items(),
        colors,
    ):
        h = res["accuracy_history"]
        ax.plot(range(len(h)), h, label=name, color=color, lw=2, marker="o", ms=4)

    axes[0].set_title("Per-Chunk Accuracy: Tree vs Forest")
    axes[0].set_xlabel("Chunk")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Bar chart: loop vs vectorised ---
    bars = axes[1].bar(
        ["Loop", "Vectorised"],
        [loop_time * 1000, vec_time * 1000],
        color=["#DC2626", "#16A34A"],
        width=0.4,
    )
    axes[1].set_title("Gini Scoring: Loop vs NumPy Vectorised")
    axes[1].set_ylabel("Time (ms)")
    for bar in bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{bar.get_height():.1f} ms",
            ha="center", va="bottom", fontsize=10,
        )
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "benchmark_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    model_results = benchmark_model_comparison()
    loop_time, vec_time, speedup = benchmark_gini_vectorised_vs_loop()
    plot_benchmark_results(model_results, loop_time, vec_time)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, res in model_results.items():
        print(f"  {name:<35} CumAcc={res['cumulative_accuracy']:.4f}")
    print(f"  Vectorised speedup: {speedup:.2f}×")