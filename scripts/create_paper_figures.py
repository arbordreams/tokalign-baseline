"""
Create publication-quality figures for SRW paper.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style for publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colors
BLUE = '#2563eb'
RED = '#dc2626'
GREEN = '#16a34a'
GRAY = '#6b7280'
ORANGE = '#ea580c'

def create_fertility_trap_figure():
    """
    Main figure showing the Fertility Trap paradox.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    
    # Data
    metrics = ['Baseline', 'Adapted']
    
    # Panel A: Fertility (GOOD)
    ax = axes[0]
    fertility = [2.059, 1.874]
    colors = [GRAY, GREEN]
    bars = ax.bar(metrics, fertility, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Fertility (tokens/word)')
    ax.set_title('(a) Tokenization Efficiency', fontweight='bold')
    ax.set_ylim(0, 2.5)
    # Add percentage change
    ax.annotate('-9%', xy=(1, 1.874), xytext=(1, 2.1),
                ha='center', fontsize=11, color=GREEN, fontweight='bold')
    ax.annotate('Improved', xy=(1, 1.95), ha='center', fontsize=9, color=GREEN)
    
    # Panel B: Throughput (BAD)
    ax = axes[1]
    throughput = [619.77, 496.22]
    colors = [GRAY, RED]
    bars = ax.bar(metrics, throughput, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Throughput (tokens/sec)')
    ax.set_title('(b) Computational Efficiency', fontweight='bold')
    ax.set_ylim(0, 750)
    ax.annotate('-20%', xy=(1, 496.22), xytext=(1, 580),
                ha='center', fontsize=11, color=RED, fontweight='bold')
    ax.annotate('Degraded', xy=(1, 540), ha='center', fontsize=9, color=RED)
    
    # Panel C: chrF-E (BAD)
    ax = axes[2]
    chrfe = [33.31, 30.18]
    colors = [GRAY, RED]
    bars = ax.bar(metrics, chrfe, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('chrF-E Score')
    ax.set_title('(c) Efficiency-Weighted Quality', fontweight='bold')
    ax.set_ylim(0, 40)
    ax.annotate('-9.4%', xy=(1, 30.18), xytext=(1, 35),
                ha='center', fontsize=11, color=RED, fontweight='bold')
    ax.annotate('Net Loss', xy=(1, 33), ha='center', fontsize=9, color=RED)
    
    # Add equation annotation
    fig.text(0.5, -0.08, 
             'chrF-E = chrF++ × (Throughput_adapted / Throughput_baseline) = 37.70 × (496/620) = 30.18',
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('visualizations/fertility_trap.png', bbox_inches='tight', pad_inches=0.1)
    plt.savefig('visualizations/fertility_trap.pdf', bbox_inches='tight', pad_inches=0.1)
    print("✓ Saved fertility_trap.png/pdf")
    plt.close()


def create_metric_disagreement_figure():
    """
    Figure showing BLEU vs chrF++ vs COMET disagreement.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    metrics = ['BLEU', 'COMET', 'chrF++']
    changes = [-12.0, -9.6, +13.2]
    colors = [RED, RED, GREEN]
    
    bars = ax.barh(metrics, changes, color=colors, edgecolor='black', linewidth=0.5, height=0.6)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, changes)):
        if val > 0:
            ax.text(val + 0.5, i, f'+{val}%', va='center', fontsize=11, fontweight='bold')
        else:
            ax.text(val - 0.5, i, f'{val}%', va='center', ha='right', fontsize=11, fontweight='bold')
    
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Change from Baseline (%)')
    ax.set_title('Metric Disagreement: BLEU & COMET vs chrF++', fontweight='bold')
    ax.set_xlim(-20, 20)
    
    # Add annotation
    ax.annotate('COMET (neural)\nagrees with BLEU', 
                xy=(-10, 1.5), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('visualizations/metric_disagreement.png', bbox_inches='tight')
    plt.savefig('visualizations/metric_disagreement.pdf', bbox_inches='tight')
    print("✓ Saved metric_disagreement.png/pdf")
    plt.close()


def create_chrfd_figure():
    """
    Figure showing chrF-Δ with confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Data with confidence intervals
    models = ['Baseline', 'Adapted']
    chrfd = [16.66, 18.93]
    ci_low = [15.78, 18.15]
    ci_high = [17.53, 19.72]
    
    x = np.arange(len(models))
    colors = [GRAY, BLUE]
    
    # Bar plot with error bars
    bars = ax.bar(x, chrfd, color=colors, edgecolor='black', linewidth=0.5, width=0.5)
    
    # Add error bars
    yerr_low = [chrfd[i] - ci_low[i] for i in range(len(chrfd))]
    yerr_high = [ci_high[i] - chrfd[i] for i in range(len(chrfd))]
    ax.errorbar(x, chrfd, yerr=[yerr_low, yerr_high], fmt='none', 
                color='black', capsize=5, capthick=2, linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('chrF-Δ (chrF++ − BLEU)')
    ax.set_title('chrF-Δ with 95% Bootstrap CI', fontweight='bold')
    ax.set_ylim(0, 25)
    
    # Add annotation for difference
    ax.annotate('', xy=(1, 20), xytext=(0, 17.5),
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=2))
    ax.text(0.5, 21.5, 'Δ = +2.27\n[1.38, 3.14]', ha='center', fontsize=10, 
            color=BLUE, fontweight='bold')
    ax.text(0.5, 24, 'p < 0.05', ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig('visualizations/chrfd_ci.png', bbox_inches='tight')
    plt.savefig('visualizations/chrfd_ci.pdf', bbox_inches='tight')
    print("✓ Saved chrfd_ci.png/pdf")
    plt.close()


def create_summary_figure():
    """
    Combined summary figure for the paper.
    """
    fig = plt.figure(figsize=(10, 8))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    
    # Panel A: Metric Disagreement
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['BLEU', 'COMET', 'chrF++']
    changes = [-12.0, -9.6, +13.2]
    colors = [RED, RED, GREEN]
    bars = ax1.barh(metrics, changes, color=colors, edgecolor='black', linewidth=0.5, height=0.6)
    ax1.axvline(x=0, color='black', linewidth=1)
    ax1.set_xlabel('Change (%)')
    ax1.set_title('(a) MT Metric Disagreement', fontweight='bold')
    ax1.set_xlim(-20, 20)
    for i, val in enumerate(changes):
        label = f'+{val}%' if val > 0 else f'{val}%'
        pos = val + 1 if val > 0 else val - 1
        ha = 'left' if val > 0 else 'right'
        ax1.text(pos, i, label, va='center', ha=ha, fontsize=10, fontweight='bold')
    
    # Panel B: chrF-Δ with CI
    ax2 = fig.add_subplot(gs[0, 1])
    models = ['Baseline', 'Adapted']
    chrfd = [16.66, 18.93]
    ci_low = [15.78, 18.15]
    ci_high = [17.53, 19.72]
    x = [0, 1]
    bars = ax2.bar(x, chrfd, color=[GRAY, BLUE], edgecolor='black', linewidth=0.5, width=0.5)
    yerr_low = [chrfd[i] - ci_low[i] for i in range(2)]
    yerr_high = [ci_high[i] - chrfd[i] for i in range(2)]
    ax2.errorbar(x, chrfd, yerr=[yerr_low, yerr_high], fmt='none', 
                color='black', capsize=5, capthick=2, linewidth=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylabel('chrF-Δ')
    ax2.set_title('(b) chrF-Δ Diagnostic', fontweight='bold')
    ax2.set_ylim(0, 24)
    ax2.text(0.5, 22, 'Δ=+2.27, p<0.05', ha='center', fontsize=10, color=BLUE, fontweight='bold')
    
    # Panel C: Fertility Trap
    ax3 = fig.add_subplot(gs[1, :])
    
    categories = ['Fertility\n(tokens/word)', 'Throughput\n(tokens/sec)', 'chrF-E\n(quality)']
    baseline = [2.059, 619.77, 33.31]
    adapted = [1.874, 496.22, 30.18]
    
    # Normalize for comparison
    baseline_norm = [100, 100, 100]
    adapted_norm = [
        (adapted[0]/baseline[0])*100,  # 91% (good, lower is better for fertility)
        (adapted[1]/baseline[1])*100,  # 80% (bad)
        (adapted[2]/baseline[2])*100   # 91% (bad)
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, baseline_norm, width, label='Baseline', 
                    color=GRAY, edgecolor='black', linewidth=0.5)
    bars2 = ax3.bar(x + width/2, adapted_norm, width, label='Adapted',
                    color=[GREEN, RED, RED], edgecolor='black', linewidth=0.5)
    
    ax3.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Relative to Baseline (%)')
    ax3.set_title('(c) The Fertility Trap: Tokenization vs Computational Efficiency', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.set_ylim(0, 120)
    ax3.legend(loc='upper right')
    
    # Add annotations
    annotations = ['−9%\n(better)', '−20%\n(worse)', '−9.4%\n(worse)']
    colors_ann = [GREEN, RED, RED]
    for i, (ann, col) in enumerate(zip(annotations, colors_ann)):
        ax3.text(i + width/2, adapted_norm[i] + 3, ann, ha='center', 
                fontsize=9, color=col, fontweight='bold')
    
    # Add conclusion text box
    ax3.text(1, 15, 'Despite improved fertility, net efficiency-adjusted quality decreases!',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor=ORANGE, linewidth=2))
    
    plt.savefig('visualizations/paper_summary.png', bbox_inches='tight')
    plt.savefig('visualizations/paper_summary.pdf', bbox_inches='tight')
    print("✓ Saved paper_summary.png/pdf")
    plt.close()


if __name__ == '__main__':
    print("Creating publication figures...")
    create_fertility_trap_figure()
    create_metric_disagreement_figure()
    create_chrfd_figure()
    create_summary_figure()
    print("\nAll figures created successfully!")

