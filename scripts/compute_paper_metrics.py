#!/usr/bin/env python3
"""
Compute additional metrics for the chrF-Δ and chrF-E paper.
Requires: pip install unbabel-comet sacrebleu pandas numpy

This script computes:
1. COMET scores (neural MT metric trained on human judgments)
2. chrF-Δ (BLEU-chrF disagreement diagnostic)
3. chrF-E (efficiency-weighted quality)
4. Bootstrap confidence intervals for new metrics
"""

import pandas as pd
import numpy as np
from sacrebleu import corpus_bleu, corpus_chrf
import json
import os

# =============================================================================
# Configuration
# =============================================================================
RESULTS_DIR = "results"
OUTPUT_FILE = "results/paper_metrics.json"

# Your existing results (from RESULTS_SUMMARY.md)
BASELINE_METRICS = {
    "bleu": 16.25,
    "chrf": 33.31,
    "ter": 83.07,
    "throughput": 619.77,  # tokens/sec
    "fertility": 2.059,
}

ADAPTED_METRICS = {
    "bleu": 14.30,
    "chrf": 37.70,
    "ter": 101.18,
    "throughput": 496.22,  # tokens/sec
    "fertility": 1.874,
}

# =============================================================================
# Compute chrF-Δ (Tokenization Bias Diagnostic)
# =============================================================================
def compute_chrf_delta(chrf: float, bleu: float) -> float:
    """
    chrF-Δ = chrF++ - BLEU
    
    Measures disagreement between character-level and token-level metrics.
    Large positive shift after vocabulary adaptation indicates BLEU is 
    penalizing tokenization changes, not quality degradation.
    """
    return chrf - bleu


print("=" * 60)
print("COMPUTING PAPER METRICS")
print("=" * 60)

# Compute chrF-Δ
baseline_chrf_delta = compute_chrf_delta(BASELINE_METRICS["chrf"], BASELINE_METRICS["bleu"])
adapted_chrf_delta = compute_chrf_delta(ADAPTED_METRICS["chrf"], ADAPTED_METRICS["bleu"])
delta_chrf_delta = adapted_chrf_delta - baseline_chrf_delta

print(f"\n--- chrF-Δ (Tokenization Bias Diagnostic) ---")
print(f"Baseline chrF-Δ: {baseline_chrf_delta:.2f}")
print(f"Adapted chrF-Δ:  {adapted_chrf_delta:.2f}")
print(f"Δ chrF-Δ:        {delta_chrf_delta:+.2f}")
print(f"\nInterpretation: +{delta_chrf_delta:.2f} point shift indicates BLEU is")
print(f"underestimating quality improvement due to tokenization sensitivity.")

# =============================================================================
# Compute chrF-E (Efficiency-Weighted Quality)
# =============================================================================
def compute_chrf_e(chrf: float, throughput: float, baseline_throughput: float) -> float:
    """
    chrF-E = chrF++ × (throughput / baseline_throughput)
    
    Weights translation quality by computational efficiency.
    Reveals the "Fertility Trap" where quality improves but efficiency degrades.
    """
    return chrf * (throughput / baseline_throughput)


baseline_chrf_e = compute_chrf_e(
    BASELINE_METRICS["chrf"], 
    BASELINE_METRICS["throughput"], 
    BASELINE_METRICS["throughput"]
)
adapted_chrf_e = compute_chrf_e(
    ADAPTED_METRICS["chrf"], 
    ADAPTED_METRICS["throughput"], 
    BASELINE_METRICS["throughput"]
)
chrf_e_change = (adapted_chrf_e - baseline_chrf_e) / baseline_chrf_e * 100

print(f"\n--- chrF-E (Efficiency-Weighted Quality) ---")
print(f"Baseline chrF-E: {baseline_chrf_e:.2f}")
print(f"Adapted chrF-E:  {adapted_chrf_e:.2f}")
print(f"Change:          {chrf_e_change:+.1f}%")
print(f"\nFertility Trap Analysis:")
print(f"  chrF++ improved:  +{(ADAPTED_METRICS['chrf'] - BASELINE_METRICS['chrf']) / BASELINE_METRICS['chrf'] * 100:.1f}%")
print(f"  Throughput change: {(ADAPTED_METRICS['throughput'] - BASELINE_METRICS['throughput']) / BASELINE_METRICS['throughput'] * 100:.1f}%")
print(f"  Net chrF-E change: {chrf_e_change:+.1f}%")
print(f"\n  → FERTILITY TRAP DETECTED: Quality improved but efficiency-adjusted quality decreased!")

# =============================================================================
# Compute COMET (if available)
# =============================================================================
comet_baseline = None
comet_adapted = None

try:
    from comet import download_model, load_from_checkpoint
    
    print(f"\n--- COMET (Neural MT Metric) ---")
    print("Loading COMET model...")
    
    # Load model
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    
    # Load MT results
    baseline_mt = pd.read_csv(f"{RESULTS_DIR}/mt_results_baseline.csv")
    adapted_mt = pd.read_csv(f"{RESULTS_DIR}/mt_results_adapted.csv")
    
    print(f"Computing COMET for {len(baseline_mt)} baseline translations...")
    baseline_data = [
        {"src": row['source'], "mt": row['hypothesis'], "ref": row['reference']}
        for _, row in baseline_mt.iterrows()
        if pd.notna(row['hypothesis']) and row['hypothesis'].strip()
    ]
    baseline_output = model.predict(baseline_data, batch_size=8, gpus=1)
    comet_baseline = baseline_output.system_score
    print(f"Baseline COMET: {comet_baseline:.4f}")
    
    print(f"Computing COMET for {len(adapted_mt)} adapted translations...")
    adapted_data = [
        {"src": row['source'], "mt": row['hypothesis'], "ref": row['reference']}
        for _, row in adapted_mt.iterrows()
        if pd.notna(row['hypothesis']) and row['hypothesis'].strip()
    ]
    adapted_output = model.predict(adapted_data, batch_size=8, gpus=1)
    comet_adapted = adapted_output.system_score
    print(f"Adapted COMET:  {comet_adapted:.4f}")
    
    comet_change = (comet_adapted - comet_baseline) / abs(comet_baseline) * 100
    print(f"Change:         {comet_change:+.1f}%")
    
    # Check if COMET agrees with chrF++ (both improve) or BLEU (both degrade)
    chrf_improved = ADAPTED_METRICS["chrf"] > BASELINE_METRICS["chrf"]
    bleu_improved = ADAPTED_METRICS["bleu"] > BASELINE_METRICS["bleu"]
    comet_improved = comet_adapted > comet_baseline
    
    print(f"\n--- Metric Agreement Analysis ---")
    print(f"chrF++ improved: {chrf_improved} ({'+' if chrf_improved else ''}{(ADAPTED_METRICS['chrf'] - BASELINE_METRICS['chrf']) / BASELINE_METRICS['chrf'] * 100:.1f}%)")
    print(f"BLEU improved:   {bleu_improved} ({'+' if bleu_improved else ''}{(ADAPTED_METRICS['bleu'] - BASELINE_METRICS['bleu']) / BASELINE_METRICS['bleu'] * 100:.1f}%)")
    print(f"COMET improved:  {comet_improved} ({'+' if comet_improved else ''}{comet_change:.1f}%)")
    
    if comet_improved == chrf_improved and comet_improved != bleu_improved:
        print(f"\n✓ COMET agrees with chrF++, not BLEU!")
        print(f"  This supports using chrF++ as the primary quality metric.")
    elif comet_improved == bleu_improved and comet_improved != chrf_improved:
        print(f"\n! COMET agrees with BLEU, not chrF++")
        print(f"  Further investigation needed.")
    else:
        print(f"\n? All metrics agree or disagree in complex pattern")
        
except ImportError:
    print(f"\n--- COMET ---")
    print("COMET not installed. Run: pip install unbabel-comet")
    print("Then re-run this script.")
except Exception as e:
    print(f"\n--- COMET ---")
    print(f"Error computing COMET: {e}")

# =============================================================================
# Bootstrap CI for chrF-Δ (per-sample)
# =============================================================================
print(f"\n--- Bootstrap CI for chrF-Δ ---")

try:
    # Load per-sample MT results to compute per-sample chrF-Δ
    baseline_mt = pd.read_csv(f"{RESULTS_DIR}/mt_results_baseline.csv")
    adapted_mt = pd.read_csv(f"{RESULTS_DIR}/mt_results_adapted.csv")
    
    from sacrebleu import sentence_bleu, sentence_chrf
    
    def compute_per_sample_metrics(df, desc=""):
        """Compute per-sample BLEU and chrF."""
        bleu_scores = []
        chrf_scores = []
        
        print(f"  Computing per-sample metrics for {desc}...")
        for _, row in df.iterrows():
            hyp = str(row['hypothesis']) if pd.notna(row['hypothesis']) else ""
            ref = str(row['reference']) if pd.notna(row['reference']) else ""
            
            if hyp.strip() and ref.strip():
                bleu = sentence_bleu(hyp, [ref]).score
                chrf = sentence_chrf(hyp, [ref]).score
            else:
                bleu = 0
                chrf = 0
            
            bleu_scores.append(bleu)
            chrf_scores.append(chrf)
        
        return np.array(bleu_scores), np.array(chrf_scores)
    
    baseline_bleu, baseline_chrf_arr = compute_per_sample_metrics(baseline_mt, "baseline")
    adapted_bleu, adapted_chrf_arr = compute_per_sample_metrics(adapted_mt, "adapted")
    
    # Compute per-sample chrF-Δ
    baseline_chrf_delta_arr = baseline_chrf_arr - baseline_bleu
    adapted_chrf_delta_arr = adapted_chrf_arr - adapted_bleu
    
    # Bootstrap
    def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
        means = [np.mean(np.random.choice(data, len(data), replace=True)) 
                 for _ in range(n_bootstrap)]
        lower = np.percentile(means, (1-ci)/2 * 100)
        upper = np.percentile(means, (1+ci)/2 * 100)
        return np.mean(data), lower, upper
    
    b_mean, b_lo, b_hi = bootstrap_ci(baseline_chrf_delta_arr)
    a_mean, a_lo, a_hi = bootstrap_ci(adapted_chrf_delta_arr)
    
    # Difference
    diff = adapted_chrf_delta_arr - baseline_chrf_delta_arr
    d_mean, d_lo, d_hi = bootstrap_ci(diff)
    
    print(f"\nBaseline chrF-Δ: {b_mean:.2f} [{b_lo:.2f}, {b_hi:.2f}]")
    print(f"Adapted chrF-Δ:  {a_mean:.2f} [{a_lo:.2f}, {a_hi:.2f}]")
    print(f"Difference:      {d_mean:.2f} [{d_lo:.2f}, {d_hi:.2f}]")
    
    if d_lo > 0:
        print(f"\n✓ chrF-Δ increase is statistically significant (95% CI excludes 0)")
    else:
        print(f"\n! chrF-Δ increase may not be statistically significant")
        
except Exception as e:
    print(f"Error computing bootstrap CI: {e}")

# =============================================================================
# Save Results
# =============================================================================
results = {
    "chrf_delta": {
        "baseline": baseline_chrf_delta,
        "adapted": adapted_chrf_delta,
        "delta": delta_chrf_delta,
    },
    "chrf_e": {
        "baseline": baseline_chrf_e,
        "adapted": adapted_chrf_e,
        "change_percent": chrf_e_change,
    },
    "comet": {
        "baseline": comet_baseline,
        "adapted": comet_adapted,
    },
    "fertility_trap": {
        "chrf_improved_percent": (ADAPTED_METRICS['chrf'] - BASELINE_METRICS['chrf']) / BASELINE_METRICS['chrf'] * 100,
        "throughput_change_percent": (ADAPTED_METRICS['throughput'] - BASELINE_METRICS['throughput']) / BASELINE_METRICS['throughput'] * 100,
        "chrf_e_change_percent": chrf_e_change,
        "trap_detected": chrf_e_change < 0,
    }
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 60}")
print(f"Results saved to: {OUTPUT_FILE}")
print(f"{'=' * 60}")

# =============================================================================
# LaTeX Table Output
# =============================================================================
print(f"\n--- LaTeX Table (copy to paper) ---")
print(r"""
\begin{table}[h]
\centering
\caption{Proposed metrics reveal evaluation dynamics missed by traditional metrics.}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{Adapted} & \textbf{$\Delta$} \\
\midrule
BLEU & 16.25 & 14.30 & $-12.0\%$ \\
chrF++ & 33.31 & 37.70 & $+13.2\%$ \\""")

if comet_baseline and comet_adapted:
    comet_change = (comet_adapted - comet_baseline) / abs(comet_baseline) * 100
    print(f"COMET & {comet_baseline:.3f} & {comet_adapted:.3f} & ${comet_change:+.1f}\\%$ \\\\")

print(r"""\midrule
chrF-$\Delta$ & """ + f"{baseline_chrf_delta:.2f} & {adapted_chrf_delta:.2f} & $+{delta_chrf_delta:.2f}$ \\\\" + r"""
chrF-E & """ + f"{baseline_chrf_e:.2f} & {adapted_chrf_e:.2f} & ${chrf_e_change:+.1f}\\%$ \\\\" + r"""
\bottomrule
\end{tabular}
\end{table}
""")

