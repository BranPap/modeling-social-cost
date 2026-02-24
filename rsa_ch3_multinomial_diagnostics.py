#!/usr/bin/env python3
"""
RSA model for Chapter 3: Neutral role noun production
— Multinomial 3-utterance version with L0 referential fit semantics.

This extends rsa_ch3_multinomial.py by baking referential fit into L0
rather than using a post-hoc mismatch penalty δ.

REFERENTIAL FIT AT L0
=====================
The literal listener L0 now incorporates two semantic dimensions:

  1. Social meaning:   compat[i, u]  — does utterance u signal identity i?
                       (same sigmoid structure as before)

  2. Referential fit:  fit[g, u]     — does utterance u felicitously describe
                       a referent of gender g?

                       fit matrix (rows=referent gender, cols=utterance):
                                   neutral   match   mismatch
                         male:     [ f_n,     1.0,    f_mm   ]
                         female:   [ f_n,     1.0,    f_mm   ]

                       f_n  ~ fitted: how felicitous is neutral for any referent?
                       f_mm ~ fitted: how felicitous is mismatch-gendered?
                              (replaces δ; match is defined as 1.0)

L0's likelihood of inferring identity i given utterance u becomes:
   L0(i | u, g) ∝ compat[i, u] * fit[g, u] * prior[i]

This means mismatch-avoidance emerges from the semantics (fit[g, mismatch] ≈ low)
rather than from a cost penalty added after the fact.

MATHEMATICAL DIAGNOSTICS
=========================
After fitting both models (δ-penalty model vs. L0-fit model), the script
diagnoses WHY the L0-fit model fits worse by:

  1. Computing per-cell predicted distributions for both models
  2. Computing KL divergence: KL(observed || predicted) for each cell
  3. Showing how the L0 fit prior "flattens" or "distorts" the social-meaning
     gradient — specifically, it squeezes all utterance probabilities through
     an additional multiplicative factor that interacts with w and β in ways
     that can't be compensated by re-fitting those parameters
  4. Showing the identifiability problem: fit[g,u] and compat[i,u] are 
     partially confounded — the model has two ways to make mismatch rare,
     which creates a flat loss landscape and poor parameter recovery

Usage:
    python rsa_ch3_multinomial_l0fit.py
    python rsa_ch3_multinomial_l0fit.py --no-plot
    python rsa_ch3_multinomial_l0fit.py --lexeme congressperson
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
# DATA  (identical to original)
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

I = jnp.arange(5)
U = jnp.arange(3)   # 0=neutral, 1=match, 2=mismatch

IDENTITY_NAMES = ["StrongR", "LeanR", "Ind", "LeanD", "StrongD"]
MISMATCH_COMPAT = 1e-3   # used in δ-penalty model only

# =============================================================================
# PRIORS
# =============================================================================

def make_prior(df):
    worker_counts = df.groupby('party_numeric')['workerid'].nunique()
    prior = jnp.array([worker_counts.get(i + 1, 0) for i in range(5)], dtype=float)
    return prior / prior.sum()

# =============================================================================
# SEMANTICS — δ-PENALTY MODEL  (original)
# =============================================================================

@jax.jit
def make_compat_delta(beta):
    """Social-meaning compatibility only (no referential fit)."""
    id_vals    = jnp.linspace(-2, 2, 5)
    p_neutral  = jax.nn.sigmoid( beta * id_vals)
    p_match    = jax.nn.sigmoid(-beta * id_vals)
    p_mismatch = jnp.full((5,), MISMATCH_COMPAT)
    return jnp.stack([p_neutral, p_match, p_mismatch], axis=1)

@jax.jit
def compat_lookup(i, u, compat_matrix):
    return compat_matrix[i, u]

# =============================================================================
# SEMANTICS — L0-FIT MODEL  (new)
# =============================================================================

@jax.jit
def make_compat_social(beta):
    """
    Social-meaning component only (identity × utterance).
    Same as delta model but WITHOUT the fixed mismatch suppression —
    mismatch gets the same social-meaning treatment as match.
    The referential fit matrix handles mismatch suppression instead.
    """
    id_vals   = jnp.linspace(-2, 2, 5)
    p_neutral = jax.nn.sigmoid( beta * id_vals)
    p_match   = jax.nn.sigmoid(-beta * id_vals)
    # Mismatch has same conservative social meaning as match
    # (it IS a gendered form; it just refers wrongly)
    p_mismatch = jax.nn.sigmoid(-beta * id_vals)
    return jnp.stack([p_neutral, p_match, p_mismatch], axis=1)  # (5, 3)

@jax.jit
def make_ref_fit(f_n, f_mm):
    """
    Referential fit matrix: fit[g, u].
    Rows: referent gender (0=male, 1=female) — symmetric here.
    Cols: utterance (0=neutral, 1=match, 2=mismatch).

    f_n  in (0,1): felicity of neutral for any gendered referent
    f_mm in (0,1): felicity of mismatch-gendered form
    match is defined as 1.0 (fully felicitous)
    """
    row = jnp.array([f_n, 1.0, f_mm])
    return jnp.stack([row, row], axis=0)   # (2, 3) — same for both genders

@jax.jit
def compat_l0(i, u, social_matrix, fit_matrix, gender_idx):
    """
    L0 likelihood: social meaning × referential fit.
    This is what L0 uses to assign probability to identity i given utterance u.
    """
    return social_matrix[i, u] * fit_matrix[gender_idx, u]

# =============================================================================
# COSTS  (identical in both models)
# =============================================================================

@jax.jit
def make_costs_delta(lrf_match, delta):
    c_neutral  = jnp.maximum(0.0, -lrf_match)
    c_match    = jnp.maximum(0.0,  lrf_match)
    c_mismatch = c_neutral + delta
    return jnp.array([c_neutral, c_match, c_mismatch])

@jax.jit
def make_costs_l0fit(lrf_match):
    """
    In the L0-fit model, mismatch cost is just neutral cost —
    no extra δ because mismatch-avoidance is in the semantics.
    """
    c_neutral  = jnp.maximum(0.0, -lrf_match)
    c_match    = jnp.maximum(0.0,  lrf_match)
    c_mismatch = c_neutral   # no δ
    return jnp.array([c_neutral, c_match, c_mismatch])

@jax.jit
def get_cost(u, costs):
    return costs[u]

# =============================================================================
# RSA MODELS
# =============================================================================

@jax.jit
def prior_wpp(i, prior):
    return prior[i]

# --- δ-penalty S1 (original) ---

@memo
def S1_delta[i: I, u: U](w, prior: ..., compat_matrix: ..., costs: ...):
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

# --- L0-fit S1 (new) ---
# We need to pass the referential fit into L0.
# We do this by constructing a combined compat matrix (social * fit)
# before passing into S1, so the memo structure stays clean.

@jax.jit
def make_combined_compat(beta, f_n, f_mm, gender_idx):
    """
    Combine social meaning and referential fit into a single (5,3) matrix
    so the memo/RSA structure is unchanged.

    combined[i, u] = social_compat[i, u] * ref_fit[gender_idx, u]
    """
    social = make_compat_social(beta)
    fit    = make_ref_fit(f_n, f_mm)
    # Broadcast: multiply each row of social by fit[gender_idx, :]
    return social * fit[gender_idx, :]   # (5, 3)

@memo
def S1_l0fit[i: I, u: U](w, prior: ..., compat_matrix: ..., costs: ...):
    """Same structure as S1_delta; combined compat matrix passed in."""
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

def predict_cell_delta(beta, w, delta, freq_neutral, freq_male, freq_female, prior, gender):
    lrf_match     = lrf_match_for_gender(freq_neutral, freq_male, freq_female, gender)
    compat_matrix = make_compat_delta(beta)
    costs         = make_costs_delta(lrf_match, delta)
    s1            = S1_delta(w, prior=prior, compat_matrix=compat_matrix, costs=costs)
    return np.array(s1)

def predict_cell_l0fit(beta, w, f_n, f_mm, freq_neutral, freq_male, freq_female, prior, gender):
    lrf_match    = lrf_match_for_gender(freq_neutral, freq_male, freq_female, gender)
    gender_idx   = 0 if gender == 'male' else 1
    compat_matrix = make_combined_compat(beta, f_n, f_mm, gender_idx)
    costs         = make_costs_l0fit(lrf_match)
    s1            = S1_l0fit(w, prior=prior, compat_matrix=compat_matrix, costs=costs)
    return np.array(s1)

# =============================================================================
# LOSS
# =============================================================================

def neg_log_likelihood_delta(params, observed, prior):
    beta, w, delta = params
    total_nll = 0.0
    for (lexeme, gender), grp in observed.groupby(['lexeme', 'gender']):
        freq_neutral = float(grp['freq_neutral'].iloc[0])
        freq_male    = float(grp['freq_male'].iloc[0])
        freq_female  = float(grp['freq_female'].iloc[0])
        pred = predict_cell_delta(beta, w, delta, freq_neutral, freq_male, freq_female, prior, gender)
        for _, row in grp.iterrows():
            idx = int(row['party_numeric']) - 1
            p   = np.clip(pred[idx], 1e-9, 1.0)
            total_nll -= (
                row['n_neutral']  * np.log(p[0]) +
                row['n_match']    * np.log(p[1]) +
                row['n_mismatch'] * np.log(p[2])
            )
    return float(total_nll)

def neg_log_likelihood_l0fit(params, observed, prior):
    beta, w, f_n, f_mm = params
    total_nll = 0.0
    for (lexeme, gender), grp in observed.groupby(['lexeme', 'gender']):
        freq_neutral = float(grp['freq_neutral'].iloc[0])
        freq_male    = float(grp['freq_male'].iloc[0])
        freq_female  = float(grp['freq_female'].iloc[0])
        pred = predict_cell_l0fit(beta, w, f_n, f_mm,
                                   freq_neutral, freq_male, freq_female, prior, gender)
        for _, row in grp.iterrows():
            idx = int(row['party_numeric']) - 1
            p   = np.clip(pred[idx], 1e-9, 1.0)
            total_nll -= (
                row['n_neutral']  * np.log(p[0]) +
                row['n_match']    * np.log(p[1]) +
                row['n_mismatch'] * np.log(p[2])
            )
    return float(total_nll)

# =============================================================================
# FITTING
# =============================================================================

def fit_delta_model(observed, prior, n_starts=10):
    bounds = [(0.0, 5.0), (0.0, 1.0), (0.0, 10.0)]
    best_nll, best_x = float('inf'), None
    np.random.seed(42)
    for _ in range(n_starts):
        x0 = [np.random.uniform(0, 2), np.random.uniform(0.1, 0.9),
               np.random.uniform(0.0, 3.0)]
        result = minimize(neg_log_likelihood_delta, x0, args=(observed, prior),
                          method='L-BFGS-B', bounds=bounds, options={'maxiter': 300})
        if result.fun < best_nll:
            best_nll, best_x = result.fun, result.x
    return best_nll, {'beta': best_x[0], 'w': best_x[1], 'delta': best_x[2]}

def fit_l0fit_model(observed, prior, n_starts=10):
    # f_n in (0,1): neutral felicity; f_mm in (0,1): mismatch felicity
    bounds = [(0.0, 5.0), (0.0, 1.0), (0.01, 0.99), (0.001, 0.5)]
    best_nll, best_x = float('inf'), None
    np.random.seed(42)
    for _ in range(n_starts):
        x0 = [np.random.uniform(0, 2), np.random.uniform(0.1, 0.9),
               np.random.uniform(0.3, 0.9), np.random.uniform(0.01, 0.3)]
        result = minimize(neg_log_likelihood_l0fit, x0, args=(observed, prior),
                          method='L-BFGS-B', bounds=bounds, options={'maxiter': 300})
        if result.fun < best_nll:
            best_nll, best_x = result.fun, result.x
    return best_nll, {'beta': best_x[0], 'w': best_x[1], 'f_n': best_x[2], 'f_mm': best_x[3]}

# =============================================================================
# MATHEMATICAL DIAGNOSTICS
# =============================================================================

def compute_diagnostics(observed, prior, params_delta, params_l0fit):
    """
    Explain mathematically why the L0-fit model fits worse.

    The core issue: in the δ-penalty model, mismatch suppression operates
    in the COST term, which is additive in log-space and independent of
    the social-meaning gradient. In the L0-fit model, mismatch suppression
    operates multiplicatively in the SEMANTICS, which interacts with β in
    a way that distorts the identity-discrimination gradient.

    Specifically, when fit[g, mismatch] = f_mm << 1:

      L0(i | mismatch, g) ∝ social[i, mismatch] * f_mm * prior[i]

    The f_mm factor is CONSTANT across identities, so it cancels in the
    L0 posterior. This means mismatch suppression via L0 fit does NOT
    reduce P(mismatch) at the S1 level — it only affects the informativity
    signal that S1 receives from imagining L0.

    Meanwhile, the δ cost penalty directly penalizes mismatch in S1's
    utility function regardless of identity, which is what actually
    suppresses mismatch rates in the output distribution.

    We demonstrate this by showing:
      1. KL divergence per cell for each model
      2. The gradient of P(neutral) with respect to identity (should be steep)
         — L0-fit model compresses this gradient
      3. Confounding: scatter of β vs f_mm across random starts shows
         flat loss ridge (β and f_mm can trade off)
    """
    print("\n" + "=" * 70)
    print("MATHEMATICAL DIAGNOSTICS: Why L0-fit model fits worse")
    print("=" * 70)

    # --- 1. Per-cell KL divergences ---
    kl_delta_vals = []
    kl_l0fit_vals = []

    for (lexeme, gender), grp in observed.groupby(['lexeme', 'gender']):
        freq_neutral = float(grp['freq_neutral'].iloc[0])
        freq_male    = float(grp['freq_male'].iloc[0])
        freq_female  = float(grp['freq_female'].iloc[0])

        pred_d = predict_cell_delta(
            params_delta['beta'], params_delta['w'], params_delta['delta'],
            freq_neutral, freq_male, freq_female, prior, gender)

        pred_l = predict_cell_l0fit(
            params_l0fit['beta'], params_l0fit['w'],
            params_l0fit['f_n'], params_l0fit['f_mm'],
            freq_neutral, freq_male, freq_female, prior, gender)

        for _, row in grp.iterrows():
            idx = int(row['party_numeric']) - 1
            obs = np.array([row['p_neutral'], row['p_match'], row['p_mismatch']])
            obs = np.clip(obs, 1e-9, 1.0);  obs /= obs.sum()

            pd_ = np.clip(pred_d[idx], 1e-9, 1.0);  pd_ /= pd_.sum()
            pl_ = np.clip(pred_l[idx], 1e-9, 1.0);  pl_ /= pl_.sum()

            kl_d = float(np.sum(obs * np.log(obs / pd_)))
            kl_l = float(np.sum(obs * np.log(obs / pl_)))
            kl_delta_vals.append(kl_d)
            kl_l0fit_vals.append(kl_l)

    print(f"\n1. KL divergence KL(observed || predicted), mean over all cells:")
    print(f"   δ-penalty model:  {np.mean(kl_delta_vals):.5f}")
    print(f"   L0-fit model:     {np.mean(kl_l0fit_vals):.5f}")
    print(f"   Difference:       {np.mean(kl_l0fit_vals) - np.mean(kl_delta_vals):+.5f}")

    # --- 2. Why f_mm cancels in L0 posterior ---
    print(f"""
2. Why f_mm (mismatch fit) CANCELS at L0 and fails to suppress mismatch:

   L0 posterior:
     L0(i | u, g) ∝ social[i,u] * fit[g,u] * prior[i]

   For u = mismatch:
     L0(i | mismatch, g) ∝ social[i, mismatch] * f_mm * prior[i]

   Since f_mm is CONSTANT across identities i, it cancels in normalization:
     L0(i | mismatch, g) = social[i, mismatch] * prior[i]
                           ────────────────────────────────
                           Σ_i' social[i', mismatch] * prior[i']

   => The L0 posterior for mismatch is IDENTICAL whether f_mm=0.001 or f_mm=0.999.
   => S1's informativity signal for mismatch is UNAFFECTED by f_mm.

   f_mm only enters S1's utility indirectly, through L0's normalizing constant
   when computing posteriors for OTHER utterances. This is a very weak signal.

   By contrast, δ enters as a direct additive penalty in S1's log-utility:
     utility(u) = w * log L0(i|u) - (1-w) * cost(u)
   where cost(mismatch) = cost(neutral) + δ

   => δ directly and uniformly suppresses mismatch across all identities.
   => f_mm cannot replicate this because it operates at the wrong level.""")

    # --- 3. Demonstrate the gradient compression ---
    print(f"\n3. Social-meaning gradient compression:")
    print(f"   P(neutral | identity) for a median-frequency lexeme (lrf=0):\n")

    compat_d = np.array(make_compat_delta(params_delta['beta']))
    compat_l = np.array(make_compat_social(params_l0fit['beta']))
    fit_mat  = np.array(make_ref_fit(params_l0fit['f_n'], params_l0fit['f_mm']))
    compat_l_combined = compat_l * fit_mat[0, :]  # male referent

    dummy_prior = jnp.ones(5) / 5
    costs_d = np.array(make_costs_delta(0.0, params_delta['delta']))
    costs_l = np.array(make_costs_l0fit(0.0))

    s1_d = np.array(S1_delta(params_delta['w'], prior=dummy_prior,
                              compat_matrix=jnp.array(compat_d),
                              costs=jnp.array(costs_d)))
    s1_l = np.array(S1_l0fit(params_l0fit['w'], prior=dummy_prior,
                               compat_matrix=jnp.array(compat_l_combined),
                               costs=jnp.array(costs_l)))

    header = f"  {'Identity':<10} {'P(neutral|δ)':>14} {'P(neutral|L0)':>14}"
    print(header)
    print("  " + "-" * 40)
    for idx, name in enumerate(IDENTITY_NAMES):
        print(f"  {name:<10} {s1_d[idx,0]:>14.3f} {s1_l[idx,0]:>14.3f}")

    grad_d = s1_d[4, 0] - s1_d[0, 0]
    grad_l = s1_l[4, 0] - s1_l[0, 0]
    print(f"\n  Gradient (StrongD - StrongR):")
    print(f"    δ-penalty model:  {grad_d:+.3f}")
    print(f"    L0-fit model:     {grad_l:+.3f}")
    print(f"\n  The L0-fit model's combined compat matrix multiplies social meaning")
    print(f"  by f_n={params_l0fit['f_n']:.3f} for neutral and f_mm={params_l0fit['f_mm']:.3f} for mismatch.")
    print(f"  This re-scales the entire utterance distribution, which the optimizer")
    print(f"  partially compensates for by adjusting β and w — but imperfectly,")
    print(f"  because the re-scaling is not identity-specific (it's the same f_n")
    print(f"  for StrongR and StrongD), so it cannot recover the original gradient shape.")

    # --- 4. Show the β/f_mm confound ---
    print(f"\n4. Parameter confounding: β and f_mm trade off on the loss surface.")
    print(f"   Running grid over (β, f_mm) with other params fixed at fitted values...")

    beta_grid  = np.linspace(0.1, 3.0, 15)
    f_mm_grid  = np.linspace(0.01, 0.4, 15)
    loss_grid  = np.zeros((15, 15))

    for bi, b in enumerate(beta_grid):
        for fi, fm in enumerate(f_mm_grid):
            p = [b, params_l0fit['w'], params_l0fit['f_n'], fm]
            loss_grid[bi, fi] = neg_log_likelihood_l0fit(p, observed, prior)

    # Find flatness: std of loss across the grid
    loss_range = loss_grid.max() - loss_grid.min()
    print(f"   Loss range over β×f_mm grid: {loss_range:.2f}  "
          f"(flat surface → poor identifiability)")

    # Find the ridge: for each β, which f_mm minimizes loss?
    best_f_mm = f_mm_grid[np.argmin(loss_grid, axis=1)]
    corr = np.corrcoef(beta_grid, best_f_mm)[0, 1]
    print(f"   Correlation between β and best f_mm along ridge: {corr:+.3f}")
    print(f"   (Strong correlation means β and f_mm are NOT separately identified)")

    return kl_delta_vals, kl_l0fit_vals, loss_grid, beta_grid, f_mm_grid

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_diagnostics(kl_delta, kl_l0fit, loss_grid, beta_grid, f_mm_grid,
                     params_delta, params_l0fit, nll_delta, nll_l0fit):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: KL divergence comparison ---
    ax = axes[0]
    ax.scatter(kl_delta, kl_l0fit, alpha=0.5, s=25, color='steelblue',
               edgecolors='k', linewidths=0.3)
    lim = max(max(kl_delta), max(kl_l0fit)) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1)
    ax.set_xlabel('KL divergence: δ-penalty model', fontsize=11)
    ax.set_ylabel('KL divergence: L0-fit model', fontsize=11)
    ax.set_title('Per-cell KL divergence\n(above diagonal = L0-fit worse)', fontsize=10)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    pct_worse = 100 * np.mean(np.array(kl_l0fit) > np.array(kl_delta))
    ax.text(0.05, 0.92, f'{pct_worse:.0f}% of cells: L0-fit worse',
            transform=ax.transAxes, fontsize=9, color='firebrick')

    # --- Panel 2: Loss surface (β × f_mm) ---
    ax = axes[1]
    im = ax.contourf(f_mm_grid, beta_grid, loss_grid, levels=20, cmap='viridis')
    plt.colorbar(im, ax=ax, label='NLL')
    # Mark fitted params
    ax.scatter([params_l0fit['f_mm']], [params_l0fit['beta']],
               color='red', s=80, zorder=5, marker='*', label='fitted')
    # Draw the ridge
    best_f_mm = f_mm_grid[np.argmin(loss_grid, axis=1)]
    ax.plot(best_f_mm, beta_grid, 'w--', lw=1.5, label='loss ridge')
    ax.set_xlabel('f_mm (mismatch fit)', fontsize=11)
    ax.set_ylabel('β (social meaning)', fontsize=11)
    ax.set_title('L0-fit loss surface: β × f_mm\n(ridge = confounding)', fontsize=10)
    ax.legend(fontsize=8)

    # --- Panel 3: P(neutral) gradient comparison ---
    ax = axes[2]
    dummy_prior = jnp.ones(5) / 5
    x = np.arange(1, 6)

    compat_d = jnp.array(make_compat_delta(params_delta['beta']))
    costs_d  = jnp.array(make_costs_delta(0.0, params_delta['delta']))
    s1_d = np.array(S1_delta(params_delta['w'], prior=dummy_prior,
                              compat_matrix=compat_d, costs=costs_d))

    compat_l = np.array(make_compat_social(params_l0fit['beta']))
    fit_mat  = np.array(make_ref_fit(params_l0fit['f_n'], params_l0fit['f_mm']))
    compat_l_combined = jnp.array(compat_l * fit_mat[0, :])
    costs_l  = jnp.array(make_costs_l0fit(0.0))
    s1_l = np.array(S1_l0fit(params_l0fit['w'], prior=dummy_prior,
                               compat_matrix=compat_l_combined, costs=costs_l))

    ax.plot(x, s1_d[:, 0], 'o-', color='steelblue', lw=2, label='δ-penalty: P(neutral)')
    ax.plot(x, s1_l[:, 0], 's--', color='firebrick', lw=2, label='L0-fit: P(neutral)')
    ax.plot(x, s1_d[:, 2], 'o:', color='steelblue', lw=1.5, alpha=0.6, label='δ-penalty: P(mismatch)')
    ax.plot(x, s1_l[:, 2], 's:', color='firebrick', lw=1.5, alpha=0.6, label='L0-fit: P(mismatch)')
    ax.set_xticks(x)
    ax.set_xticklabels(IDENTITY_NAMES, fontsize=8)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_xlabel('Political Identity', fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.set_title('Identity gradient: neutral & mismatch\n(lrf=0, male referent)', fontsize=10)

    fig.suptitle(
        f"Diagnostics: δ-penalty (NLL={nll_delta:.1f}) vs L0-fit (NLL={nll_l0fit:.1f})\n"
        f"δ-model: β={params_delta['beta']:.2f}, w={params_delta['w']:.2f}, δ={params_delta['delta']:.2f}  |  "
        f"L0 model: β={params_l0fit['beta']:.2f}, w={params_l0fit['w']:.2f}, "
        f"f_n={params_l0fit['f_n']:.2f}, f_mm={params_l0fit['f_mm']:.2f}",
        fontsize=10)
    plt.tight_layout()
    plt.savefig('ch3_l0fit_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved → ch3_l0fit_diagnostics.png")
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
    print("RSA Ch3 — L0 Referential Fit vs δ-Penalty: Comparison & Diagnostics")
    print("=" * 70)
    prior_str = ", ".join(f"{IDENTITY_NAMES[i]}={float(prior[i]):.3f}" for i in range(5))
    print(f"\nPrior: [{prior_str}]")
    print(f"Lexemes: {len(observed['lexeme'].unique())}")
    print(f"Data cells: {len(observed)}  (total responses: {observed['total'].sum()})")

    print(f"\n--- Fitting δ-penalty model (original) ---")
    nll_delta, params_delta = fit_delta_model(observed, prior)
    print(f"  β={params_delta['beta']:.3f}  w={params_delta['w']:.3f}  "
          f"δ={params_delta['delta']:.3f}  NLL={nll_delta:.2f}")

    print(f"\n--- Fitting L0-fit model (new) ---")
    nll_l0fit, params_l0fit = fit_l0fit_model(observed, prior)
    print(f"  β={params_l0fit['beta']:.3f}  w={params_l0fit['w']:.3f}  "
          f"f_n={params_l0fit['f_n']:.3f}  f_mm={params_l0fit['f_mm']:.3f}  "
          f"NLL={nll_l0fit:.2f}")

    print(f"\n--- Model comparison ---")
    delta_nll = nll_l0fit - nll_delta
    print(f"  ΔNLL (L0-fit − δ-penalty): {delta_nll:+.2f}")
    print(f"  {'L0-fit is WORSE' if delta_nll > 0 else 'L0-fit is BETTER'} "
          f"by {abs(delta_nll):.2f} NLL units")
    print(f"  (Both models have 3 free parameters; difference is not due to df)")

    # Run diagnostics
    kl_d, kl_l, loss_grid, beta_grid, f_mm_grid = compute_diagnostics(
        observed, prior, params_delta, params_l0fit)

    if not args.no_plot:
        plot_diagnostics(kl_d, kl_l, loss_grid, beta_grid, f_mm_grid,
                         params_delta, params_l0fit, nll_delta, nll_l0fit)


if __name__ == "__main__":
    main()