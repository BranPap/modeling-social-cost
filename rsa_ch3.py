#!/usr/bin/env python3
"""
RSA model for Chapter 3: Neutral role noun production.

Models neutral role noun selection (e.g., congressperson vs congressman/woman)
as a function of political identity and lexical frequency.

Speakers choose between neutral and gendered role nouns to:
  1. Signal political identity (informativity: progressive → neutral)
  2. Minimize cost (frequency: rare forms are costly to produce)

Parameters:
  β (beta):  social meaning — how strongly neutral indexes progressiveness
  w:         mixture weight — balance between informativity (1) and cost (0)

Usage:
    python rsa_ch3.py
    python rsa_ch3.py --lexeme congressperson
"""

from memo import memo
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import argparse

jax.config.update("jax_enable_x64", True)

# =============================================================================
# DATA
# =============================================================================

DATA_PATH = "small_production_data.csv"

def load_data(path=DATA_PATH):
    return pd.read_csv(path)

def get_observed(df):
    """Observed neutral rates by party_numeric × lexeme."""
    return df.groupby(['party_numeric', 'lexeme']).agg(
        neutral_rate=('neutral_binary', 'mean'),
        n=('neutral_binary', 'count'),
        log_rel_freq=('log_rel_freq', 'first')
    ).reset_index()

# =============================================================================
# DOMAINS
# =============================================================================

I = jnp.arange(5)  # political identity: 0..4 → party_numeric 1..5
U = jnp.arange(2)  # utterance: 0=gendered, 1=neutral

IDENTITY_NAMES = ["StrongR", "LeanR", "Ind", "LeanD", "StrongD"]

# =============================================================================
# PRIORS
# =============================================================================

def make_prior(df):
    """Prior over identities from participant proportions."""
    worker_counts = df.groupby('party_numeric')['workerid'].nunique()
    prior = jnp.array([worker_counts.get(i + 1, 0) for i in range(5)], dtype=float)
    return prior / prior.sum()

# =============================================================================
# SEMANTICS
# =============================================================================

@jax.jit
def make_compat(beta):
    """
    Soft compatibility: identity × utterance.

    beta controls how strongly neutral indexes progressive identity.
    compat[i, 0] = base rate of gendered for identity i
    compat[i, 1] = base rate of neutral for identity i
    """
    id_vals = jnp.linspace(-2, 2, 5)  # centered at 0
    p_neutral = jax.nn.sigmoid(beta * id_vals)
    return jnp.stack([1 - p_neutral, p_neutral], axis=1)

@jax.jit
def compat_lookup(i, u, compat_matrix):
    return compat_matrix[i, u]

# =============================================================================
# COSTS
# =============================================================================

@jax.jit
def make_costs(log_rel_freq):
    """
    Frequency-based cost.

    log_rel_freq = log(freq_neutral / freq_gendered)
      negative → neutral is rare (costly to produce)
      positive → gendered is rare (costly to produce)
    """
    return jnp.array([
        jnp.maximum(0.0, log_rel_freq),   # cost of gendered
        jnp.maximum(0.0, -log_rel_freq),   # cost of neutral
    ])

@jax.jit
def get_cost(u, costs):
    return costs[u]

# =============================================================================
# RSA MODEL
# =============================================================================

@jax.jit
def prior_wpp(i, prior):
    return prior[i]

@memo
def S1[i: I, u: U](w, prior: ..., compat_matrix: ..., costs: ...):
    """S1 speaker: choose utterance to signal political identity."""
    speaker: knows(i)
    speaker: thinks[
        listener: thinks[
            speaker: given(i in I, wpp=prior_wpp(i, prior)),
            speaker: chooses(u in U, wpp=compat_lookup(i, u, compat_matrix))
        ]
    ]
    speaker: chooses(u in U, wpp=exp(imagine[
        listener: observes [speaker.u] is u,
        listener: knows(i),
        (
            w * listener[ log(Pr[speaker.i == i]) ] -
            (1 - w) * get_cost(u, costs)
        )
    ]))
    return Pr[speaker.u == u]

# =============================================================================
# PREDICTION
# =============================================================================

def predict_lexeme(beta, w, log_rel_freq, prior):
    """Predict P(neutral | identity) for a single lexeme."""
    compat_matrix = make_compat(beta)
    costs = make_costs(log_rel_freq)
    s1 = S1(w, prior=prior, compat_matrix=compat_matrix, costs=costs)
    return np.array(s1[:, 1])  # P(neutral) for each identity

# =============================================================================
# FITTING
# =============================================================================

def total_rmse(params, observed, prior):
    """RMSE across all party × lexeme cells."""
    beta, w = params
    errors = []
    for lexeme in observed['lexeme'].unique():
        lex_data = observed[observed['lexeme'] == lexeme]
        lrf = lex_data['log_rel_freq'].iloc[0]
        pred = predict_lexeme(beta, w, lrf, prior)
        for _, row in lex_data.iterrows():
            idx = int(row['party_numeric']) - 1
            errors.append((pred[idx] - row['neutral_rate']) ** 2)
    return float(np.sqrt(np.mean(errors)))

def fit_model(observed, prior, n_starts=10):
    """Fit β, w by minimizing RMSE."""
    bounds = [(0.0, 5), (0.0, 1.0)]
    best_rmse = float('inf')
    best_x = None

    np.random.seed(42)
    for _ in range(n_starts):
        x0 = [np.random.uniform(0, 2),
               np.random.uniform(0.1, 0.9)]
        result = minimize(total_rmse, x0, args=(observed, prior),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 300})
        if result.fun < best_rmse:
            best_rmse = result.fun
            best_x = result.x

    return best_rmse, {
        'beta': best_x[0], 'w': best_x[1]
    }

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(plot_df, params):
    """Two plots: faceted by lexeme + overall scatter."""
    lexemes = sorted(plot_df['lexeme'].unique(),
                     key=lambda l: plot_df[plot_df['lexeme'] == l]['log_rel_freq'].iloc[0])
    n = len(lexemes)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    party_colors = ['#E31A1C', '#FB9A99', '#999999', '#A6CEE3', '#1F78B4']

    # --- Plot 1: Faceted by lexeme ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i, lexeme in enumerate(lexemes):
        ax = axes[i]
        ld = plot_df[plot_df['lexeme'] == lexeme].sort_values('party_numeric')
        lrf = ld['log_rel_freq'].iloc[0]

        for j, (_, row) in enumerate(ld.iterrows()):
            ax.scatter(row['party_numeric'], row['obs'],
                       color=party_colors[j], s=50, zorder=3, edgecolors='k', linewidths=0.5)
        ax.plot(ld['party_numeric'], ld['pred'], color='black', linewidth=1.5, zorder=2)

        ax.set_title(f"{lexeme}\nlrf={lrf:.1f}", fontsize=9)
        ax.set_xticks(range(1, 6))
        if i >= (nrows - 1) * ncols:
            ax.set_xticklabels(['SR', 'LR', 'I', 'LD', 'SD'], fontsize=7)
        else:
            ax.set_xticklabels([])
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.5, color='grey', linewidth=0.5, linestyle='--', alpha=0.3)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.supxlabel('Political Identity', fontsize=11)
    fig.supylabel('P(neutral)', fontsize=11)
    fig.suptitle(f"RSA Ch3: Pred (line) vs Obs (dots)\n"
                 f"β={params['beta']:.2f}  w={params['w']:.2f}",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('ch3_faceted.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved faceted plot to ch3_faceted.png")
    plt.close()

    # --- Plot 2: Scatter of observed vs predicted ---
    fig, ax = plt.subplots(figsize=(7, 7))

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color='grey', linewidth=1, linestyle='--', zorder=1)

    for lexeme in lexemes:
        ld = plot_df[plot_df['lexeme'] == lexeme].sort_values('party_numeric')
        for j, (_, row) in enumerate(ld.iterrows()):
            ax.scatter(row['obs'], row['pred'],
                       color=party_colors[j], s=40, zorder=3,
                       edgecolors='k', linewidths=0.3, alpha=0.8)

        # Label each lexeme at its mean position
        mean_obs = ld['obs'].mean()
        mean_pred = ld['pred'].mean()
        ax.annotate(lexeme, (mean_obs, mean_pred),
                    fontsize=7, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points', alpha=0.7)

    # Compute R²
    r2 = np.corrcoef(plot_df['obs'], plot_df['pred'])[0, 1] ** 2
    rmse_val = np.sqrt(np.mean((plot_df['obs'] - plot_df['pred']) ** 2))

    ax.set_xlabel('Observed P(neutral)', fontsize=12)
    ax.set_ylabel('Predicted P(neutral)', fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_title(f"RSA Ch3: Observed vs Predicted\n"
                 f"R²={r2:.3f}  RMSE={rmse_val:.3f}", fontsize=12)

    # Legend for party colors
    for j, name in enumerate(['StrongR', 'LeanR', 'Ind', 'LeanD', 'StrongD']):
        ax.scatter([], [], color=party_colors[j], s=40, edgecolors='k',
                   linewidths=0.3, label=name)
    ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig('ch3_scatter.png', dpi=150, bbox_inches='tight')
    print(f"Saved scatter plot to ch3_scatter.png")
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexeme", type=str, default=None,
                        help="Focus on a single lexeme")
    parser.add_argument("--data", type=str, default=DATA_PATH)
    args = parser.parse_args()

    df = load_data(args.data)
    observed = get_observed(df)
    prior = make_prior(df)

    if args.lexeme:
        observed = observed[observed['lexeme'] == args.lexeme]
        if observed.empty:
            available = sorted(df['lexeme'].unique())
            print(f"Error: lexeme '{args.lexeme}' not found.")
            print(f"Available: {', '.join(available)}")
            return

    print("\n" + "=" * 70)
    print("RSA Ch3: Neutral Role Noun Production")
    print("=" * 70)
    prior_str = ", ".join(f"{IDENTITY_NAMES[i]}={float(prior[i]):.3f}"
                          for i in range(5))
    print(f"\nPrior: [{prior_str}]")
    print(f"Lexemes: {len(observed['lexeme'].unique())}")
    print(f"Data cells: {len(observed)}")

    avg_rmse, params = fit_model(observed, prior)

    print(f"\nFitted parameters:")
    print(f"  β (social meaning) = {params['beta']:.3f}")
    print(f"  w (info vs cost)   = {params['w']:.3f}  [1=pure info, 0=pure cost]")
    print(f"  RMSE               = {avg_rmse:.4f}")

    # Per-lexeme table: pred/obs for each identity
    header = f"{'Lexeme':<20} {'logFreq':>7}"
    for name in IDENTITY_NAMES:
        header += f"  {name:>11}"
    print(f"\n{header}")
    print("-" * len(header))

    # Collect results for plotting
    plot_data = []

    for lexeme in sorted(observed['lexeme'].unique()):
        lex_data = observed[observed['lexeme'] == lexeme]
        lrf = lex_data['log_rel_freq'].iloc[0]
        pred = predict_lexeme(params['beta'], params['w'], lrf, prior)
        obs_dict = {int(r['party_numeric']) - 1: r['neutral_rate']
                    for _, r in lex_data.iterrows()}

        row = f"{lexeme:<20} {lrf:>7.2f}"
        for idx in range(5):
            p = pred[idx]
            o = obs_dict.get(idx, float('nan'))
            row += f"  {p:.2f}/{o:.2f}"
            plot_data.append({
                'lexeme': lexeme, 'log_rel_freq': lrf,
                'identity': IDENTITY_NAMES[idx], 'party_numeric': idx + 1,
                'pred': p, 'obs': o,
            })
        print(row)

    # Highlight congressperson
    print(f"\n--- Congressperson detail ---")
    cp_data = observed[observed['lexeme'] == 'congressperson']
    if not cp_data.empty:
        lrf = cp_data['log_rel_freq'].iloc[0]
        pred = predict_lexeme(params['beta'], params['w'], lrf, prior)
        for _, row in cp_data.iterrows():
            idx = int(row['party_numeric']) - 1
            print(f"  {IDENTITY_NAMES[idx]:>8}: pred={pred[idx]:.3f}  "
                  f"obs={row['neutral_rate']:.3f}  (n={int(row['n'])})")

    # --- Visualization ---
    plot_df = pd.DataFrame(plot_data)
    plot_results(plot_df, params)

if __name__ == "__main__":
    main()
