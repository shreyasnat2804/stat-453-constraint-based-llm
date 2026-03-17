import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def plot_csr_degradation(metrics_by_level, model_name, results_dir):
    levels = ["L1", "L2", "L3", "L4"]
    csr_values = [metrics_by_level[l]["csr"] for l in levels]
    hard_csr_values = [metrics_by_level[l]["hard_csr"] for l in levels]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(levels, csr_values, marker="o", label="Per-Constraint CSR")
    ax.plot(levels, hard_csr_values, marker="o", label="Hard CSR")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("CSR")
    ax.set_title(f"CSR Degradation: {model_name}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(os.path.join(results_dir, f"{model_name_safe}_csr_degradation.png"))
    plt.show()


def plot_per_type_bar(per_type_rates, model_name, results_dir):
    sorted_items = sorted(per_type_rates.items(), key=lambda x: x[1], reverse=True)
    types = [item[0] for item in sorted_items]
    rates = [item[1] for item in sorted_items]
    colors = [
        "#2ecc71" if r >= 0.7 else "#f39c12" if r >= 0.4 else "#e74c3c"
        for r in rates
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(types, rates, color=colors)
    ax.set_xlim(0, 1)
    ax.set_title(f"Per-Constraint-Type Pass Rate: {model_name}")
    plt.tight_layout()

    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(os.path.join(results_dir, f"{model_name_safe}_per_type.png"))
    plt.show()


def plot_constraint_distribution(data, model_name, results_dir):
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=df,
        x="num_constraints",
        hue="difficulty_level",
        palette="viridis",
        kde=False,
        ax=ax,
    )
    ax.set_title(f"Constraint Count Distribution: {model_name}")
    plt.tight_layout()

    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(os.path.join(results_dir, f"{model_name_safe}_constraint_dist.png"))
    plt.show()
