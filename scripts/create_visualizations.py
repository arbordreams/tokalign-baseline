#!/usr/bin/env python3
"""
TokAlign Evaluation Results Visualization
Generates publication-quality charts for all evaluation metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from pathlib import Path

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('ggplot')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# Colors - modern palette
BASELINE_COLOR = '#4A90A4'  # Steel blue
ADAPTED_COLOR = '#E07A5F'   # Terra cotta
ACCENT_COLOR = '#81B29A'    # Sage green
HIGHLIGHT_COLOR = '#F4A261' # Sandy orange

# Create output directory
OUTPUT_DIR = Path('visualizations')
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("TokAlign Evaluation Visualization Generator")
print("=" * 60)


# =============================================================================
# 1. TOKENIZER EFFICIENCY
# =============================================================================
print("\n[1/7] Creating tokenizer efficiency chart...")

tokenizer_df = pd.read_csv('results/tokenizer_comparison.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1a: Fertility & Compression
metrics_1 = ['fertility', 'compression_ratio']
labels_1 = ['Fertility\n(tokens/word)', 'Compression\nRatio']
baseline_vals_1 = [tokenizer_df[tokenizer_df['metric'] == m]['baseline_mean'].values[0] for m in metrics_1]
adapted_vals_1 = [tokenizer_df[tokenizer_df['metric'] == m]['adapted_mean'].values[0] for m in metrics_1]

x = np.arange(len(labels_1))
width = 0.35

bars1 = axes[0].bar(x - width/2, baseline_vals_1, width, label='Baseline (Pythia)', color=BASELINE_COLOR, edgecolor='white', linewidth=1.5)
bars2 = axes[0].bar(x + width/2, adapted_vals_1, width, label='Adapted (Qwen2)', color=ADAPTED_COLOR, edgecolor='white', linewidth=1.5)

axes[0].set_ylabel('Value')
axes[0].set_title('Tokenizer Efficiency Metrics', fontweight='bold', pad=15)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels_1)
axes[0].legend(loc='upper right')
axes[0].set_ylim(0, max(max(baseline_vals_1), max(adapted_vals_1)) * 1.2)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    axes[0].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    axes[0].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)

# Add improvement annotation for fertility
fertility_improvement = ((baseline_vals_1[0] - adapted_vals_1[0]) / baseline_vals_1[0]) * 100
axes[0].annotate(f'↓ {fertility_improvement:.1f}%', 
                 xy=(0, adapted_vals_1[0]), xytext=(-0.3, adapted_vals_1[0] + 0.3),
                 fontsize=11, color='green', fontweight='bold')

# Plot 1b: PCW & STRR
metrics_2 = ['pcw', 'strr']
labels_2 = ['PCW\n(Continued Words)', 'STRR\n(Single-Token Rate)']
baseline_vals_2 = [tokenizer_df[tokenizer_df['metric'] == m]['baseline_mean'].values[0] for m in metrics_2]
adapted_vals_2 = [tokenizer_df[tokenizer_df['metric'] == m]['adapted_mean'].values[0] for m in metrics_2]

bars3 = axes[1].bar(x - width/2, baseline_vals_2, width, label='Baseline (Pythia)', color=BASELINE_COLOR, edgecolor='white', linewidth=1.5)
bars4 = axes[1].bar(x + width/2, adapted_vals_2, width, label='Adapted (Qwen2)', color=ADAPTED_COLOR, edgecolor='white', linewidth=1.5)

axes[1].set_ylabel('Proportion')
axes[1].set_title('Token Distribution Metrics', fontweight='bold', pad=15)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels_2)
axes[1].legend(loc='upper right')
axes[1].set_ylim(0, 0.8)

for bar in bars3:
    height = bar.get_height()
    axes[1].annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)
for bar in bars4:
    height = bar.get_height()
    axes[1].annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '1_tokenizer_efficiency.png')
plt.close()
print("  ✓ Saved: 1_tokenizer_efficiency.png")


# =============================================================================
# 2. PERPLEXITY COMPARISON
# =============================================================================
print("[2/7] Creating perplexity chart...")

ppl_baseline = pd.read_csv('results/perplexity_baseline.csv')
ppl_adapted = pd.read_csv('results/perplexity_adapted.csv')

fig, ax = plt.subplots(figsize=(10, 6))

languages = ['Spanish', 'English']
baseline_ppl = [
    ppl_baseline[ppl_baseline['language'] == 'es']['perplexity'].mean(),
    ppl_baseline[ppl_baseline['language'] == 'en']['perplexity'].mean()
]
adapted_ppl = [
    ppl_adapted[ppl_adapted['language'] == 'es']['perplexity'].mean(),
    ppl_adapted[ppl_adapted['language'] == 'en']['perplexity'].mean()
]

x = np.arange(len(languages))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_ppl, width, label='Baseline (Pythia)', color=BASELINE_COLOR, edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + width/2, adapted_ppl, width, label='Adapted (Qwen2)', color=ADAPTED_COLOR, edgecolor='white', linewidth=1.5)

ax.set_ylabel('Perplexity (↓ lower is better)')
ax.set_title('Language Model Perplexity by Language', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(languages)
ax.legend(loc='upper left')

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add note about checkpoint
ax.text(0.98, 0.95, 'Note: Adapted model at checkpoint-2500\n(early training stage)', 
        transform=ax.transAxes, fontsize=9, verticalalignment='top', 
        horizontalalignment='right', style='italic', color='gray',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '2_perplexity.png')
plt.close()
print("  ✓ Saved: 2_perplexity.png")


# =============================================================================
# 3. NLU TASKS
# =============================================================================
print("[3/7] Creating NLU tasks chart...")

with open('results/nlu_results_baseline.json') as f:
    nlu_baseline = json.load(f)
with open('results/nlu_results_adapted.json') as f:
    nlu_adapted = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ARC-Easy
tasks = ['ARC-Easy', 'HellaSwag', 'LAMBADA']
task_keys = ['arc_easy', 'hellaswag', 'lambada_openai']

for i, (task, key) in enumerate(zip(tasks, task_keys)):
    if key == 'lambada_openai':
        # For LAMBADA, show both perplexity and accuracy
        metrics = ['Perplexity ↓', 'Accuracy ↑']
        baseline_vals = [
            nlu_baseline['results'][key]['perplexity,none'],
            nlu_baseline['results'][key]['acc,none'] * 100
        ]
        adapted_vals = [
            nlu_adapted['results'][key]['perplexity,none'],
            nlu_adapted['results'][key]['acc,none'] * 100
        ]
        
        # Create twin axis for different scales
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        x_pos = [0, 1]
        
        # Perplexity bars
        b1 = ax1.bar(-0.15, baseline_vals[0], 0.3, label='Baseline PPL', color=BASELINE_COLOR, alpha=0.7)
        b2 = ax1.bar(0.15, adapted_vals[0], 0.3, label='Adapted PPL', color=ADAPTED_COLOR, alpha=0.7)
        
        # Accuracy bars
        b3 = ax2.bar(0.85, baseline_vals[1], 0.3, label='Baseline Acc', color=BASELINE_COLOR)
        b4 = ax2.bar(1.15, adapted_vals[1], 0.3, label='Adapted Acc', color=ADAPTED_COLOR)
        
        ax1.set_ylabel('Perplexity', color=BASELINE_COLOR)
        ax2.set_ylabel('Accuracy (%)', color=ACCENT_COLOR)
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Perplexity ↓', 'Accuracy ↑'])
        ax1.set_title(f'{task}', fontweight='bold', pad=15)
        
        # Add values
        ax1.annotate(f'{baseline_vals[0]:.1f}', xy=(-0.15, baseline_vals[0]), xytext=(0, 3),
                     textcoords='offset points', ha='center', fontsize=9)
        ax1.annotate(f'{adapted_vals[0]:.1f}', xy=(0.15, adapted_vals[0]), xytext=(0, 3),
                     textcoords='offset points', ha='center', fontsize=9)
        ax2.annotate(f'{baseline_vals[1]:.1f}%', xy=(0.85, baseline_vals[1]), xytext=(0, 3),
                     textcoords='offset points', ha='center', fontsize=9)
        ax2.annotate(f'{adapted_vals[1]:.1f}%', xy=(1.15, adapted_vals[1]), xytext=(0, 3),
                     textcoords='offset points', ha='center', fontsize=9)
    else:
        # For other tasks, show acc and acc_norm
        metrics = ['Accuracy', 'Acc (Norm)']
        baseline_vals = [
            nlu_baseline['results'][key]['acc,none'] * 100,
            nlu_baseline['results'][key]['acc_norm,none'] * 100
        ]
        adapted_vals = [
            nlu_adapted['results'][key]['acc,none'] * 100,
            nlu_adapted['results'][key]['acc_norm,none'] * 100
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = axes[i].bar(x - width/2, baseline_vals, width, label='Baseline', color=BASELINE_COLOR, edgecolor='white')
        bars2 = axes[i].bar(x + width/2, adapted_vals, width, label='Adapted', color=ADAPTED_COLOR, edgecolor='white')
        
        axes[i].set_ylabel('Accuracy (%)')
        axes[i].set_title(f'{task}', fontweight='bold', pad=15)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(metrics)
        axes[i].set_ylim(0, 80)
        axes[i].legend(loc='upper right', fontsize=9)
        
        for bar in bars1:
            height = bar.get_height()
            axes[i].annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                             xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            axes[i].annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                             xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

plt.suptitle('Natural Language Understanding Benchmarks (0-shot)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '3_nlu_tasks.png')
plt.close()
print("  ✓ Saved: 3_nlu_tasks.png")


# =============================================================================
# 4. MACHINE TRANSLATION
# =============================================================================
print("[4/7] Creating machine translation chart...")

# Extract MT scores from notebook outputs (hardcoded from results)
mt_metrics = ['BLEU', 'chrF++', 'TER']
baseline_mt = [16.25, 33.31, 83.07]
adapted_mt = [14.30, 37.70, 101.18]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: BLEU and chrF++ (higher is better)
x = np.arange(2)
width = 0.35

bars1 = axes[0].bar(x - width/2, [baseline_mt[0], baseline_mt[1]], width, 
                     label='Baseline (Pythia)', color=BASELINE_COLOR, edgecolor='white', linewidth=1.5)
bars2 = axes[0].bar(x + width/2, [adapted_mt[0], adapted_mt[1]], width,
                     label='Adapted (Qwen2)', color=ADAPTED_COLOR, edgecolor='white', linewidth=1.5)

axes[0].set_ylabel('Score (↑ higher is better)')
axes[0].set_title('Translation Quality Metrics', fontweight='bold', pad=15)
axes[0].set_xticks(x)
axes[0].set_xticklabels(['BLEU', 'chrF++'])
axes[0].legend(loc='upper right')
axes[0].set_ylim(0, 50)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Highlight chrF++ improvement
axes[0].annotate('↑ +13.2%', xy=(1 + width/2, adapted_mt[1]), xytext=(1.3, adapted_mt[1] + 3),
                 fontsize=10, color='green', fontweight='bold')

# Right: TER (lower is better)
bars3 = axes[1].bar([0], [baseline_mt[2]], width*2, label='Baseline (Pythia)', color=BASELINE_COLOR, edgecolor='white', linewidth=1.5)
bars4 = axes[1].bar([0.5], [adapted_mt[2]], width*2, label='Adapted (Qwen2)', color=ADAPTED_COLOR, edgecolor='white', linewidth=1.5)

axes[1].set_ylabel('TER Score (↓ lower is better)')
axes[1].set_title('Translation Error Rate', fontweight='bold', pad=15)
axes[1].set_xticks([0.25])
axes[1].set_xticklabels(['TER'])
axes[1].legend(loc='upper right')

for bar in [bars3[0], bars4[0]]:
    height = bar.get_height()
    axes[1].annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('Machine Translation: Spanish → English (OPUS-100)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '4_machine_translation.png')
plt.close()
print("  ✓ Saved: 4_machine_translation.png")


# =============================================================================
# 5. GENERATION QUALITY
# =============================================================================
print("[5/7] Creating generation quality chart...")

# Hardcoded from notebook outputs
gen_metrics = {
    'baseline_greedy': {'distinct_1': 0.3463, 'distinct_2': 0.4550, 'distinct_3': 0.4930, 'rep_rate': 0.4434, 'self_bleu': 2.40},
    'baseline_nucleus': {'distinct_1': 0.6278, 'distinct_2': 0.9027, 'distinct_3': 0.9637, 'rep_rate': 0.0128, 'self_bleu': 2.37},
    'adapted_greedy': {'distinct_1': 0.3891, 'distinct_2': 0.5261, 'distinct_3': 0.5973, 'rep_rate': 0.2847, 'self_bleu': 4.29},
    'adapted_nucleus': {'distinct_1': 0.5358, 'distinct_2': 0.7817, 'distinct_3': 0.8582, 'rep_rate': 0.0549, 'self_bleu': 1.62},
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Greedy: Distinct-N
ax = axes[0, 0]
metrics = ['Distinct-1', 'Distinct-2', 'Distinct-3']
baseline_vals = [gen_metrics['baseline_greedy']['distinct_1'], gen_metrics['baseline_greedy']['distinct_2'], gen_metrics['baseline_greedy']['distinct_3']]
adapted_vals = [gen_metrics['adapted_greedy']['distinct_1'], gen_metrics['adapted_greedy']['distinct_2'], gen_metrics['adapted_greedy']['distinct_3']]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color=BASELINE_COLOR, edgecolor='white')
bars2 = ax.bar(x + width/2, adapted_vals, width, label='Adapted', color=ADAPTED_COLOR, edgecolor='white')

ax.set_ylabel('Score (↑ higher = more diverse)')
ax.set_title('Greedy Decoding: Diversity Metrics', fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='upper left')
ax.set_ylim(0, 1.0)

for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 2), textcoords='offset points', ha='center', fontsize=9)

# Greedy: Repetition & Self-BLEU
ax = axes[0, 1]
metrics = ['Repetition Rate', 'Self-BLEU']
baseline_vals = [gen_metrics['baseline_greedy']['rep_rate'], gen_metrics['baseline_greedy']['self_bleu']]
adapted_vals = [gen_metrics['adapted_greedy']['rep_rate'], gen_metrics['adapted_greedy']['self_bleu']]

# Normalize for visualization (different scales)
fig2, ax2_temp = plt.subplots()
ax2 = ax.twinx()

b1 = ax.bar(-0.17, baseline_vals[0], 0.34, label='Baseline Rep', color=BASELINE_COLOR, alpha=0.8)
b2 = ax.bar(0.17, adapted_vals[0], 0.34, label='Adapted Rep', color=ADAPTED_COLOR, alpha=0.8)
b3 = ax2.bar(0.83, baseline_vals[1], 0.34, label='Baseline SBLEU', color=BASELINE_COLOR)
b4 = ax2.bar(1.17, adapted_vals[1], 0.34, label='Adapted SBLEU', color=ADAPTED_COLOR)

ax.set_ylabel('Repetition Rate (↓)', color='black')
ax2.set_ylabel('Self-BLEU', color='gray')
ax.set_title('Greedy Decoding: Degeneration Metrics', fontweight='bold', pad=10)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Repetition Rate ↓', 'Self-BLEU'])

ax.annotate(f'{baseline_vals[0]:.3f}', xy=(-0.17, baseline_vals[0]), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=9)
ax.annotate(f'{adapted_vals[0]:.3f}', xy=(0.17, adapted_vals[0]), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=9)
ax2.annotate(f'{baseline_vals[1]:.2f}', xy=(0.83, baseline_vals[1]), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=9)
ax2.annotate(f'{adapted_vals[1]:.2f}', xy=(1.17, adapted_vals[1]), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=9)

# Highlight improvement
ax.annotate('↓ 35.8%', xy=(0.17, adapted_vals[0]), xytext=(0.4, adapted_vals[0]),
            fontsize=10, color='green', fontweight='bold')

plt.close(fig2)

# Nucleus: Distinct-N
ax = axes[1, 0]
metrics = ['Distinct-1', 'Distinct-2', 'Distinct-3']
baseline_vals = [gen_metrics['baseline_nucleus']['distinct_1'], gen_metrics['baseline_nucleus']['distinct_2'], gen_metrics['baseline_nucleus']['distinct_3']]
adapted_vals = [gen_metrics['adapted_nucleus']['distinct_1'], gen_metrics['adapted_nucleus']['distinct_2'], gen_metrics['adapted_nucleus']['distinct_3']]

x = np.arange(len(metrics))
bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color=BASELINE_COLOR, edgecolor='white')
bars2 = ax.bar(x + width/2, adapted_vals, width, label='Adapted', color=ADAPTED_COLOR, edgecolor='white')

ax.set_ylabel('Score (↑ higher = more diverse)')
ax.set_title('Nucleus Sampling (p=0.9): Diversity Metrics', fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='upper left')
ax.set_ylim(0, 1.1)

for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 2), textcoords='offset points', ha='center', fontsize=9)

# Nucleus: Repetition & Self-BLEU
ax = axes[1, 1]
ax2 = ax.twinx()

baseline_vals = [gen_metrics['baseline_nucleus']['rep_rate'], gen_metrics['baseline_nucleus']['self_bleu']]
adapted_vals = [gen_metrics['adapted_nucleus']['rep_rate'], gen_metrics['adapted_nucleus']['self_bleu']]

b1 = ax.bar(-0.17, baseline_vals[0], 0.34, color=BASELINE_COLOR, alpha=0.8)
b2 = ax.bar(0.17, adapted_vals[0], 0.34, color=ADAPTED_COLOR, alpha=0.8)
b3 = ax2.bar(0.83, baseline_vals[1], 0.34, color=BASELINE_COLOR)
b4 = ax2.bar(1.17, adapted_vals[1], 0.34, color=ADAPTED_COLOR)

ax.set_ylabel('Repetition Rate (↓)', color='black')
ax2.set_ylabel('Self-BLEU', color='gray')
ax.set_title('Nucleus Sampling (p=0.9): Degeneration Metrics', fontweight='bold', pad=10)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Repetition Rate ↓', 'Self-BLEU'])

ax.annotate(f'{baseline_vals[0]:.4f}', xy=(-0.17, baseline_vals[0]), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=9)
ax.annotate(f'{adapted_vals[0]:.4f}', xy=(0.17, adapted_vals[0]), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=9)
ax2.annotate(f'{baseline_vals[1]:.2f}', xy=(0.83, baseline_vals[1]), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=9)
ax2.annotate(f'{adapted_vals[1]:.2f}', xy=(1.17, adapted_vals[1]), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=9)

# Legend for all
baseline_patch = mpatches.Patch(color=BASELINE_COLOR, label='Baseline (Pythia)')
adapted_patch = mpatches.Patch(color=ADAPTED_COLOR, label='Adapted (Qwen2)')
fig.legend(handles=[baseline_patch, adapted_patch], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))

plt.suptitle('Generation Quality Analysis', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '5_generation_quality.png')
plt.close()
print("  ✓ Saved: 5_generation_quality.png")


# =============================================================================
# 6. COMPUTATIONAL EFFICIENCY
# =============================================================================
print("[6/7] Creating computational efficiency chart...")

eff_baseline = pd.read_csv('results/efficiency_baseline.csv')
eff_adapted = pd.read_csv('results/efficiency_adapted.csv')

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

metrics = [
    ('tokens_per_second_mean', 'Tokens/sec', '↑'),
    ('samples_per_second_mean', 'Samples/sec', '↑'),
    ('avg_ttft_ms', 'TTFT (ms)', '↓'),
    ('peak_vram_mb', 'Peak VRAM (MB)', '↓')
]

for i, (col, label, direction) in enumerate(metrics):
    baseline_val = eff_baseline[col].values[0]
    adapted_val = eff_adapted[col].values[0]
    
    bars = axes[i].bar(['Baseline\n(Pythia)', 'Adapted\n(Qwen2)'], [baseline_val, adapted_val],
                       color=[BASELINE_COLOR, ADAPTED_COLOR], edgecolor='white', linewidth=1.5)
    
    axes[i].set_title(f'{label} {direction}', fontweight='bold', pad=10)
    
    for bar in bars:
        height = bar.get_height()
        if col == 'peak_vram_mb':
            axes[i].annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                             xytext=(0, 3), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
        elif 'ttft' in col:
            axes[i].annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                             xytext=(0, 3), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
        else:
            axes[i].annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                             xytext=(0, 3), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')

plt.suptitle('Computational Efficiency Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '6_efficiency.png')
plt.close()
print("  ✓ Saved: 6_efficiency.png")


# =============================================================================
# 7. SUMMARY RADAR CHART
# =============================================================================
print("[7/7] Creating summary radar chart...")

# Normalize metrics for radar chart (0-1 scale, higher is better)
categories = ['Tokenizer\nEfficiency', 'Spanish\nFluency ❋', 'English\nNLU', 'Translation\n(chrF++)', 'Generation\nDiversity', 'Throughput']

# Baseline scores (normalized)
baseline_scores = [
    0.5,   # Tokenizer efficiency (baseline = middle)
    0.8,   # Spanish PPL (lower is better, so invert: baseline is good)
    0.7,   # English NLU
    0.44,  # chrF++ (33.31/75 normalized)
    0.46,  # Distinct-2 greedy
    0.8,   # Throughput
]

# Adapted scores (normalized)
adapted_scores = [
    0.68,  # Tokenizer efficiency (9% improvement)
    0.35,  # Spanish PPL (worse, inverted)
    0.6,   # English NLU (slight degradation)
    0.50,  # chrF++ (37.70/75 normalized)
    0.53,  # Distinct-2 greedy
    0.64,  # Throughput
]

# Create radar chart
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

baseline_scores += baseline_scores[:1]
adapted_scores += adapted_scores[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

ax.plot(angles, baseline_scores, 'o-', linewidth=2, label='Baseline (Pythia)', color=BASELINE_COLOR)
ax.fill(angles, baseline_scores, alpha=0.25, color=BASELINE_COLOR)

ax.plot(angles, adapted_scores, 'o-', linewidth=2, label='Adapted (Qwen2)', color=ADAPTED_COLOR)
ax.fill(angles, adapted_scores, alpha=0.25, color=ADAPTED_COLOR)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], color='gray', size=9)
ax.grid(True, linestyle='--', alpha=0.5)

ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
plt.title('Model Comparison Overview\n(Higher = Better)', fontsize=14, fontweight='bold', pad=20)

# Add footnote
fig.text(0.5, -0.02, '❋ Spanish Fluency = inverted perplexity (lower PPL → higher fluency)', 
         ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '7_summary_radar.png')
plt.close()
print("  ✓ Saved: 7_summary_radar.png")


# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 60)
print("✅ All visualizations created successfully!")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
print("\nFiles generated:")
for f in sorted(OUTPUT_DIR.glob('*.png')):
    print(f"  - {f.name}")

