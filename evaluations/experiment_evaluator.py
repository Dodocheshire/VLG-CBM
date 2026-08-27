import argparse
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

def plot_anec_comparison(results_dict, dataset_name, output_path):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    colors = {
        'Standard Backbone': '#7f7f7f',
        'LF-CBM': '#1f77b4',
        'VLG-CBM (Linear)': '#ff7f0e',
        'VLG-CBM (Fine-tuned)': '#2ca02c',
        'VLG-CBM (Ours)': '#d62728'
    }
    markers = {
        'Standard Backbone': '--',
        'LF-CBM': 'o-',
        'VLG-CBM (Linear)': 's-',
        'VLG-CBM (Fine-tuned)': '^-',
        'VLG-CBM (Ours)': 'D-'
    }

    for method, data in results_dict.items():
        necs = data.get('necs', [5, 10, 15, 20, 25, 30])
        accs = [a * 100 if a <= 1.0 else a for a in data['accs']]
        color = colors.get(method, None)
        fmt = markers.get(method, 'o-')
        if method == 'Standard Backbone':
            ax.axhline(y=accs[0], color=color, linestyle='--', label=f'{method} ({accs[0]:.1f}%)', alpha=0.8)
        else:
            ax.plot(necs, accs, fmt, label=f'{method} (Avg: {np.mean(accs):.1f}%)', color=color, linewidth=2.5, markersize=8)

    ax.set_xlabel('Number of Effective Concepts (NEC)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'{dataset_name.upper()} - Concept Sparsity vs. Accuracy Curve', fontsize=15, fontweight='bold', pad=12)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='#cccccc')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'[SUCCESS] Saved publication figure to {output_path}')

def generate_comparison_table(dataset_metrics, output_dir):
    rows = []
    for ds, methods in dataset_metrics.items():
        for method_name, metrics in methods.items():
            accs = metrics.get('accs', [])
            necs = metrics.get('necs', [5, 10, 15, 20, 25, 30])
            nec_map = dict(zip(necs, accs))
            rows.append({
                'Dataset': ds.upper(),
                'Method': method_name,
                'ANEC@5 (%)': f"{nec_map.get(5, 0)*100:.2f}" if 5 in nec_map else '-',
                'ANEC@10 (%)': f"{nec_map.get(10, 0)*100:.2f}" if 10 in nec_map else '-',
                'ANEC@20 (%)': f"{nec_map.get(20, 0)*100:.2f}" if 20 in nec_map else '-',
                'ANEC@30 (%)': f"{nec_map.get(30, 0)*100:.2f}" if 30 in nec_map else '-',
                'Mean ANEC (%)': f"{np.mean(accs)*100:.2f}" if len(accs) > 0 else '-',
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
    csv_path = os.path.join(output_dir, 'tables', 'table1_benchmark_results.csv')
    md_path = os.path.join(output_dir, 'tables', 'table1_benchmark_results.md')
    df.to_csv(csv_path, index=False)
    with open(md_path, 'w') as f:
        f.write('# Benchmark Comparison Table (VLG-CBM vs Baselines)\n\n')
        f.write(df.to_markdown(index=False))
        f.write('\n')
    print(f'[SUCCESS] Saved comparison tables to {csv_path} and {md_path}')
