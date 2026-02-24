#!/usr/bin/env python3
"""
RSA model for Chapter 3: Neutral role noun production
— Multinomial 3-utterance version (gender inside S1, fit to full response distribution).

Observed data: for each participant × lexeme × referent-gender trial, they produced
  one of three forms: neutral / match-gendered / mismatch-gendered.
  This maps to freq_gender ∈ {neutral, male, female} in the dataset.

Model output: S1 predicts a full distribution over all three forms.
  s1[i, 0] = P(neutral        | identity i)
  s1[i, 1] = P(match-gendered | identity i)
  s1[i, 2] = P(mismatch       | identity i)

Loss: negative multinomial log-likelihood (cross-entropy) summed over all
  party × lexeme × gender cells, weighted by cell counts.
  This replaces the RMSE-on-P(neutral) loss and uses all three response proportions.

Utterance space per referent gender g:
  u=0  neutral           (congressperson)
  u=1  match-gendered    (congressman for male ref / congresswoman for female ref)
  u=2  mismatch-gendered (congresswoman for male ref / congressman for female ref)

Semantics:
  neutral    ~ sigmoid( β * identity)   [neutral indexes progressive]
  match      ~ sigmoid(-β * identity)   [gendered indexes conservative]
  mismatch   ~ small constant ε         [pragmatically anomalous]

Costs (referent-conditioned):
  cost(neutral)    = max(0, -lrf_match)    lrf_match = log(freq_neutral/freq_match)
  cost(match)      = max(0,  lrf_match)
  cost(mismatch)   = cost(neutral) + δ    extra penalty δ

Parameters:
  β (beta):   social meaning — how strongly neutral indexes progressive identity
  w:          mixture weight — informativity (1) vs cost (0)
  δ (delta):  mismatch penalty

Usage:
    python rsa_ch3_multinomial.py
    python rsa_ch3_multinomial.py --lexeme congressperson
    python rsa_ch3_multinomial.py --no-plot
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
EXCLUDED_LEXEMES = {'anchor'}

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    excluded = df['lexeme'].isin(EXCLUDED_LEXEMES)
    if excluded.sum():
        print(f"Excluding {excluded.sum()} rows for: {EXCLUDED_LEXEMES}")
    return df[~excluded].reset_index(drop=True)

def get_observed(df):
    """
    Observed 3-way response counts by party × lexeme × referent-gender.

    For each cell we record:
      n_neutral, n_match, n_mismatch   (raw counts)
      total                             (sum)
    
    match   = form whose gender agrees with referent gender
    mismatch = form whose gender disagrees
    """
    rows = []
    for (party, lexeme, ref_gender), grp in df.groupby(
            ['party_numeric', 'lexeme', 'gender']):

        n_neutral  = (grp['freq_gender'] == 'neutral').sum()

        if ref_gender == 'male':
            n_match    = (grp['freq_gender'] == 'male').sum()
            n_mismatch = (grp['freq_gender'] == 'female').sum()
        else:
            n_match    = (grp['freq_gender'] == 'female').sum()
            n_mismatch = (grp['freq_gender'] == 'male').sum()

        total = len(grp)
        rows.append({
            'party_numeric':  party,
            'lexeme':         lexeme,
            'gender':         ref_gender,
            'n_neutral':      int(n_neutral),
            'n_match':        int(n_match),
            'n_mismatch':     int(n_mismatch),
            'total':          int(total),
            'p_neutral':      n_neutral / total,
            'p_match':        n_match   / total,
            'p_mismatch':     n_mismatch / total,
            'freq_neutral':   float(grp['freq_neutral'].iloc[0]),
            'freq_male':      float(grp['freq_male'].iloc[0]),
            'freq_female':    float(grp['freq_female'].iloc[0]),
            'log_rel_freq':   float(grp['log_rel_freq'].iloc[0]),
        })
    return pd.DataFrame(rows)

# =============================================================================
# DOMAINS
# =============================================================================

I = jnp.arange(5)   # political identity 0..4
U = jnp.arange(3)   # 0=neutral, 1=match, 2=mismatch

IDENTITY_NAMES = ["StrongR", "LeanR", "Ind", "LeanD", "StrongD"]
MISMATCH_COMPAT = 1e-3

# =============================================================================
# PRIORS
# =============================================================================

def make_prior(df):
    worker_counts = df.groupby('party_numeric')['workerid'].nunique()
    prior = jnp.array([worker_counts.get(i + 1, 0) for i in range(5)], dtype=float)
    return prior / prior.sum()

# =============================================================================
# SEMANTICS
# =============================================================================

@jax.jit
def make_compat(beta):
    """3-way compatibility matrix (5 identities × 3 utterances)."""
    id_vals    = jnp.linspace(-2, 2, 5)
    p_neutral  = jax.nn.sigmoid( beta * id_vals)
    p_match    = jax.nn.sigmoid(-beta * id_vals)
    p_mismatch = jnp.full((5,), MISMATCH_COMPAT)
    return jnp.stack([p_neutral, p_match, p_mismatch], axis=1)

@jax.jit
def compat_lookup(i, u, compat_matrix):
    return compat_matrix[i, u]

# =============================================================================
# COSTS
# =============================================================================

@jax.jit
def make_costs(lrf_match, delta):
    """
    Referent-conditioned cost vector.
    lrf_match = log(freq_neutral / freq_match_gendered)
    """
    c_neutral  = jnp.maximum(0.0, -lrf_match)
    c_match    = jnp.maximum(0.0,  lrf_match)
    c_mismatch = c_neutral + delta
    return jnp.array([c_neutral, c_match, c_mismatch])

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
    """S1 speaker over 3-way utterance space."""
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

def lrf_match_for_gender(freq_neutral, freq_male, freq_female, gender):
    freq_match = freq_male if gender == 'male' else max(freq_female, 1)
    return float(np.log(freq_neutral / max(freq_match, 1)))

def predict_cell(beta, w, delta, freq_neutral, freq_male, freq_female, prior, gender):
    """
    Returns (5 × 3) array: P(u | identity) for all identities.
    Columns: [neutral, match, mismatch]
    """
    lrf_match     = lrf_match_for_gender(freq_neutral, freq_male, freq_female, gender)
    compat_matrix = make_compat(beta)
    costs         = make_costs(lrf_match, delta)
    s1            = S1(w, prior=prior, compat_matrix=compat_matrix, costs=costs)
    return np.array(s1)   # shape (5, 3)

# =============================================================================
# LOSS: MULTINOMIAL LOG-LIKELIHOOD
# =============================================================================

def neg_log_likelihood(params, observed, prior):
    """
    Negative multinomial log-likelihood summed over all cells,
    weighted by cell counts.

    For each party × lexeme × gender cell with counts (n0, n1, n2):
      LL += n0*log(p0) + n1*log(p1) + n2*log(p2)
    """
    beta, w, delta = params
    total_nll = 0.0
    for (lexeme, gender), grp in observed.groupby(['lexeme', 'gender']):
        freq_neutral = float(grp['freq_neutral'].iloc[0])
        freq_male    = float(grp['freq_male'].iloc[0])
        freq_female  = float(grp['freq_female'].iloc[0])
        pred = predict_cell(beta, w, delta, freq_neutral, freq_male, freq_female,
                            prior, gender)   # (5, 3)
        for _, row in grp.iterrows():
            idx = int(row['party_numeric']) - 1
            p   = np.clip(pred[idx], 1e-9, 1.0)
            total_nll -= (
                row['n_neutral']  * np.log(p[0]) +
                row['n_match']    * np.log(p[1]) +
                row['n_mismatch'] * np.log(p[2])
            )
    return float(total_nll)

def compute_rmse_neutral(params, observed, prior):
    """Also track RMSE on P(neutral) for comparability with prior models."""
    beta, w, delta = params
    errors = []
    for (lexeme, gender), grp in observed.groupby(['lexeme', 'gender']):
        freq_neutral = float(grp['freq_neutral'].iloc[0])
        freq_male    = float(grp['freq_male'].iloc[0])
        freq_female  = float(grp['freq_female'].iloc[0])
        pred = predict_cell(beta, w, delta, freq_neutral, freq_male, freq_female,
                            prior, gender)
        for _, row in grp.iterrows():
            idx = int(row['party_numeric']) - 1
            errors.append((pred[idx, 0] - row['p_neutral']) ** 2)
    return float(np.sqrt(np.mean(errors)))

# =============================================================================
# FITTING
# =============================================================================

def fit_model(observed, prior, n_starts=10):
    """Fit β, w, δ by minimising negative multinomial log-likelihood."""
    bounds = [(0.0, 5.0), (0.0, 1.0), (0.0, 10.0)]
    best_nll = float('inf')
    best_x   = None

    np.random.seed(42)
    for _ in range(n_starts):
        x0 = [np.random.uniform(0, 2),
               np.random.uniform(0.1, 0.9),
               np.random.uniform(0.0, 3.0)]
        result = minimize(neg_log_likelihood, x0, args=(observed, prior),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 300})
        if result.fun < best_nll:
            best_nll = result.fun
            best_x   = result.x

    params = {'beta': best_x[0], 'w': best_x[1], 'delta': best_x[2]}
    rmse   = compute_rmse_neutral(best_x, observed, prior)
    return best_nll, rmse, params

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(plot_df, params, nll, rmse):
    lexemes = sorted(plot_df['lexeme'].unique(),
                     key=lambda l: plot_df[plot_df['lexeme'] == l]['log_rel_freq'].mean())
    party_colors   = ['#E31A1C', '#FB9A99', '#999999', '#A6CEE3', '#1F78B4']
    gender_markers = {'male': 'o', 'female': 's'}
    gender_ls      = {'male': '-', 'female': '--'}
    utt_styles     = {
        'neutral': ('black', '-'),
        'match':   ('#444444', '--'),
        'mismatch':('#888888', ':'),
    }

    n     = len(lexemes)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    # --- Plot 1: Faceted — all 3 predicted curves + observed dots ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i, lexeme in enumerate(lexemes):
        ax  = axes[i]
        ld  = plot_df[plot_df['lexeme'] == lexeme]
        lrf = ld['log_rel_freq'].mean()

        for gender in ['male', 'female']:
            gd = ld[ld['gender'] == gender].sort_values('party_numeric')
            if gd.empty:
                continue
            # Observed dots: one per response type
            for utt_col, obs_col in [('pred_neutral','p_neutral'),
                                      ('pred_match','p_match'),
                                      ('pred_mismatch','p_mismatch')]:
                for j, (_, row) in enumerate(gd.iterrows()):
                    marker = gender_markers[gender]
                    color  = party_colors[j]
                    ax.scatter(row['party_numeric'], row[obs_col],
                               color=color, s=35, zorder=3,
                               marker=marker, edgecolors='k', linewidths=0.4,
                               alpha=0.6)
            # Predicted lines: one per utterance type
            for utt_col, label in [('pred_neutral','neutral'),
                                    ('pred_match','match'),
                                    ('pred_mismatch','mismatch')]:
                lc, base_ls = utt_styles[label]
                ls = base_ls if gender == 'male' else \
                     {'--': ':', '-': '--', ':': '-.'}[base_ls]
                ax.plot(gd['party_numeric'], gd[utt_col],
                        color=lc, linewidth=1.2, zorder=2, linestyle=ls)

        ax.set_title(f"{lexeme}\nlrf={lrf:.1f}", fontsize=8)
        ax.set_xticks(range(1, 6))
        if i >= (nrows - 1) * ncols:
            ax.set_xticklabels(['SR','LR','I','LD','SD'], fontsize=7)
        else:
            ax.set_xticklabels([])
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.5, color='grey', linewidth=0.4, linestyle=':', alpha=0.3)

    # Legend
    axes[0].plot([], [], color='black',   ls='-',  lw=1.2, label='neutral')
    axes[0].plot([], [], color='#444444', ls='--', lw=1.2, label='match')
    axes[0].plot([], [], color='#888888', ls=':',  lw=1.2, label='mismatch')
    axes[0].plot([], [], color='k', ls='-',  lw=1.2, label='male ref')
    axes[0].plot([], [], color='k', ls='--', lw=1.2, label='female ref')
    axes[0].legend(fontsize=6, loc='upper left')

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.supxlabel('Political Identity', fontsize=11)
    fig.supylabel('P(response)', fontsize=11)
    fig.suptitle(
        f"RSA Ch3 Multinomial: Pred (lines) vs Obs (dots)\n"
        f"β={params['beta']:.2f}  w={params['w']:.2f}  δ={params['delta']:.2f}  "
        f"NLL={nll:.1f}  RMSE(neutral)={rmse:.3f}",
        fontsize=10)
    plt.tight_layout()
    plt.savefig('ch3_multinomial_faceted.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved → ch3_multinomial_faceted.png")
    plt.close()

    # --- Plot 2: Scatter — observed vs predicted for ALL 3 response types ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    titles = ['P(neutral)', 'P(match-gendered)', 'P(mismatch-gendered)']
    obs_cols  = ['p_neutral',  'p_match',  'p_mismatch']
    pred_cols = ['pred_neutral','pred_match','pred_mismatch']

    for ax, title, obs_col, pred_col in zip(axes, titles, obs_cols, pred_cols):
        ax.plot([0,1],[0,1], color='grey', lw=1, ls='--', zorder=1)
        for _, row in plot_df.iterrows():
            j = int(row['party_numeric']) - 1
            ax.scatter(row[obs_col], row[pred_col],
                       color=party_colors[j], s=35,
                       marker=gender_markers[row['gender']],
                       edgecolors='k', linewidths=0.3, alpha=0.8, zorder=3)
        valid = plot_df[[obs_col, pred_col]].dropna()
        r2    = np.corrcoef(valid[obs_col], valid[pred_col])[0,1] ** 2
        rmse_ = np.sqrt(np.mean((valid[obs_col] - valid[pred_col])**2))
        ax.set_title(f"{title}\nR²={r2:.3f}  RMSE={rmse_:.3f}", fontsize=11)
        ax.set_xlabel('Observed', fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')

    axes[0].set_ylabel('Predicted', fontsize=10)

    # shared legend
    for j, name in enumerate(IDENTITY_NAMES):
        axes[2].scatter([], [], color=party_colors[j], s=35,
                        edgecolors='k', linewidths=0.3, label=name)
    axes[2].scatter([], [], color='grey', marker='o', s=35, label='male ref')
    axes[2].scatter([], [], color='grey', marker='s', s=35, label='female ref')
    axes[2].legend(fontsize=8, loc='lower right')

    fig.suptitle(
        f"RSA Ch3 Multinomial: Observed vs Predicted (all 3 response types)\n"
        f"β={params['beta']:.2f}  w={params['w']:.2f}  δ={params['delta']:.2f}",
        fontsize=12)
    plt.tight_layout()
    plt.savefig('ch3_multinomial_scatter.png', dpi=150, bbox_inches='tight')
    print(f"Saved → ch3_multinomial_scatter.png")
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexeme",  type=str,  default=None)
    parser.add_argument("--data",    type=str,  default=DATA_PATH)
    parser.add_argument("--no-plot", action='store_true')
    args = parser.parse_args()

    df       = load_data(args.data)
    observed = get_observed(df)
    prior    = make_prior(df)

    if args.lexeme:
        observed = observed[observed['lexeme'] == args.lexeme]
        if observed.empty:
            available = sorted(df['lexeme'].unique())
            print(f"Error: lexeme '{args.lexeme}' not found.")
            print(f"Available: {', '.join(available)}")
            return

    print("\n" + "=" * 70)
    print("RSA Ch3 — Multinomial 3-utterance model (gender inside S1)")
    print("=" * 70)
    print("\nLoss: negative multinomial log-likelihood over {neutral, match, mismatch}")
    prior_str = ", ".join(f"{IDENTITY_NAMES[i]}={float(prior[i]):.3f}"
                          for i in range(5))
    print(f"\nPrior: [{prior_str}]")
    print(f"Lexemes: {len(observed['lexeme'].unique())}")
    print(f"Data cells: {len(observed)}  "
          f"(total responses: {observed['total'].sum()})")

    best_nll, rmse, params = fit_model(observed, prior)

    print(f"\nFitted parameters:")
    print(f"  β  (social meaning)   = {params['beta']:.3f}")
    print(f"  w  (info vs cost)     = {params['w']:.3f}  [1=pure info, 0=pure cost]")
    print(f"  δ  (mismatch penalty) = {params['delta']:.3f}")
    print(f"  NLL                   = {best_nll:.2f}")
    print(f"  RMSE on P(neutral)    = {rmse:.4f}  [for comparison with prior models]")

    # Per-lexeme table: all 3 predicted vs observed proportions
    header = f"{'Lexeme':<20} {'g':>1}  {'lrf':>5}  "
    header += "  ".join(f"{n[:4]:>9}" for n in ['neut','match','mismt'])
    print(f"\n{header}")
    print("-" * 70)

    plot_data = []

    for lexeme in sorted(observed['lexeme'].unique()):
        for gender in ['male', 'female']:
            grp = observed[
                (observed['lexeme'] == lexeme) & (observed['gender'] == gender)
            ].sort_values('party_numeric')
            if grp.empty:
                continue

            freq_neutral = float(grp['freq_neutral'].iloc[0])
            freq_male    = float(grp['freq_male'].iloc[0])
            freq_female  = float(grp['freq_female'].iloc[0])
            lrf          = float(grp['log_rel_freq'].iloc[0])
            lrf_match    = lrf_match_for_gender(freq_neutral, freq_male, freq_female, gender)

            pred = predict_cell(params['beta'], params['w'], params['delta'],
                                freq_neutral, freq_male, freq_female, prior, gender)

            for _, row in grp.iterrows():
                idx   = int(row['party_numeric']) - 1
                p_neu = pred[idx, 0]
                p_mat = pred[idx, 1]
                p_mis = pred[idx, 2]
                plot_data.append({
                    'lexeme':       lexeme,
                    'gender':       gender,
                    'log_rel_freq': lrf,
                    'lrf_match':    lrf_match,
                    'party_numeric': idx + 1,
                    'identity':     IDENTITY_NAMES[idx],
                    'pred_neutral':  float(p_neu),
                    'pred_match':    float(p_mat),
                    'pred_mismatch': float(p_mis),
                    'p_neutral':    float(row['p_neutral']),
                    'p_match':      float(row['p_match']),
                    'p_mismatch':   float(row['p_mismatch']),
                    'n_neutral':    int(row['n_neutral']),
                    'n_match':      int(row['n_match']),
                    'n_mismatch':   int(row['n_mismatch']),
                    'total':        int(row['total']),
                })

            # Print aggregated row (mean over parties)
            g_abbr = 'm' if gender == 'male' else 'f'
            pred_mean = pred.mean(axis=0)
            obs_mean  = grp[['p_neutral','p_match','p_mismatch']].mean()
            row_str   = f"{lexeme:<20} {g_abbr}  {lrf_match:>5.2f}  "
            row_str  += (f"{pred_mean[0]:.2f}/{obs_mean['p_neutral']:.2f}  "
                         f"{pred_mean[1]:.2f}/{obs_mean['p_match']:.2f}  "
                         f"{pred_mean[2]:.2f}/{obs_mean['p_mismatch']:.2f}")
            print(row_str)

    # Congressperson detail
    print(f"\n--- Congressperson detail (pred / obs) ---")
    for gender in ['male', 'female']:
        cp = observed[
            (observed['lexeme'] == 'congressperson') & (observed['gender'] == gender)
        ].sort_values('party_numeric')
        if cp.empty:
            continue
        freq_neutral = float(cp['freq_neutral'].iloc[0])
        freq_male    = float(cp['freq_male'].iloc[0])
        freq_female  = float(cp['freq_female'].iloc[0])
        pred         = predict_cell(params['beta'], params['w'], params['delta'],
                                    freq_neutral, freq_male, freq_female, prior, gender)
        lrf_match    = lrf_match_for_gender(freq_neutral, freq_male, freq_female, gender)
        print(f"  Referent: {gender}  (lrf_match={lrf_match:.2f})")
        print(f"  {'':>8}  {'neutral':>14}  {'match':>14}  {'mismatch':>14}")
        for _, row in cp.iterrows():
            idx = int(row['party_numeric']) - 1
            print(f"  {IDENTITY_NAMES[idx]:>8}  "
                  f"{pred[idx,0]:.3f}/{row['p_neutral']:.3f}  "
                  f"{pred[idx,1]:.3f}/{row['p_match']:.3f}  "
                  f"{pred[idx,2]:.3f}/{row['p_mismatch']:.3f}  "
                  f"(n={row['total']})")

    if not args.no_plot:
        plot_df = pd.DataFrame(plot_data)
        plot_results(plot_df, params, best_nll, rmse)


if __name__ == "__main__":
    main()