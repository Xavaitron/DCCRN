"""
Compute SI-SDR, PESQ, STOI metrics for Task 1 (Anechoic).
Compares both mixture (before) and processed (after) against the target.
Outputs: metrics.json in the same directory.

Usage: python compute_metrics.py
"""
import os
import json
import numpy as np
import torchaudio
from pesq import pesq
from pystoi import stoi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RATE = 16000
SAMPLES = [1, 2, 3]
SAMPLE_LABELS = {1: "Male+Female", 2: "Male+Music", 3: "Male+Noise"}


def load_mono(path):
    """Load audio file and return as 1D numpy array (first channel)."""
    waveform, sr = torchaudio.load(path, backend="soundfile")
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    return waveform[0].numpy()


def compute_sisdr(estimate, reference):
    """Scale-Invariant Signal-to-Distortion Ratio (dB)."""
    est = estimate - np.mean(estimate)
    ref = reference - np.mean(reference)
    ref_energy = np.sum(ref ** 2) + 1e-8
    projection = np.sum(est * ref) * ref / ref_energy
    noise = est - projection
    return float(10 * np.log10(np.sum(projection ** 2) / (np.sum(noise ** 2) + 1e-8) + 1e-8))


def compute_all_metrics(estimate, reference):
    """Compute SI-SDR, PESQ, STOI. Returns dict."""
    min_len = min(len(estimate), len(reference))
    est = estimate[:min_len]
    ref = reference[:min_len]

    si_sdr = compute_sisdr(est, ref)

    try:
        pesq_score = float(pesq(SAMPLE_RATE, ref, est, 'wb'))
    except Exception:
        pesq_score = None

    try:
        stoi_score = float(stoi(ref, est, SAMPLE_RATE, extended=False))
    except Exception:
        stoi_score = None

    return {"SI-SDR": round(si_sdr, 4), "PESQ": round(pesq_score, 4) if pesq_score else None, "STOI": round(stoi_score, 4) if stoi_score else None}


def main():
    results = {}

    for s in SAMPLES:
        mixture_path = os.path.join(SCRIPT_DIR, f"mixture_signal{s}.wav")
        target_path = os.path.join(SCRIPT_DIR, f"target_signal{s}.wav")
        processed_path = os.path.join(SCRIPT_DIR, f"processed_signal{s}.wav")

        missing = [p for p in [mixture_path, target_path, processed_path] if not os.path.exists(p)]
        if missing:
            print(f"Sample {s}: skipping — missing {[os.path.basename(m) for m in missing]}")
            continue

        target = load_mono(target_path)
        mixture = load_mono(mixture_path)
        processed = load_mono(processed_path)

        mix_metrics = compute_all_metrics(mixture, target)
        proc_metrics = compute_all_metrics(processed, target)

        delta = {}
        for key in ["SI-SDR", "PESQ", "STOI"]:
            if mix_metrics[key] is not None and proc_metrics[key] is not None:
                delta[key] = round(proc_metrics[key] - mix_metrics[key], 4)
            else:
                delta[key] = None

        results[f"sample_{s}"] = {
            "label": SAMPLE_LABELS[s],
            "mixture_vs_target": mix_metrics,
            "processed_vs_target": proc_metrics,
            "improvement": delta,
        }

        print(f"Sample {s} ({SAMPLE_LABELS[s]}):")
        print(f"  Mixture  -> SI-SDR: {mix_metrics['SI-SDR']}, PESQ: {mix_metrics['PESQ']}, STOI: {mix_metrics['STOI']}")
        print(f"  Processed-> SI-SDR: {proc_metrics['SI-SDR']}, PESQ: {proc_metrics['PESQ']}, STOI: {proc_metrics['STOI']}")
        print(f"  Delta    -> SI-SDR: {delta['SI-SDR']}, PESQ: {delta['PESQ']}, STOI: {delta['STOI']}")
        print()

    output_path = os.path.join(SCRIPT_DIR, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
