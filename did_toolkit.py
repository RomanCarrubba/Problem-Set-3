"""Difference-in-differences toolkit (ECON 5200 PS3), extending Lab 11's did_utils.py."""
from __future__ import annotations
import numpy as np
import pandas as pd

NEVER = 10**9

# Lab 11's functions, carried over
def two_by_two(df: pd.DataFrame, outcome: str, treated: str, post: str) -> float:
    """The canonical 2x2: difference of differences of the four group means."""
    g = df.groupby([treated, post])[outcome].mean()
    return (g[1, 1] - g[1, 0]) - (g[0, 1] - g[0, 0])


def event_study(cells: pd.DataFrame) -> pd.DataFrame:
    """Average the ATT(g, t) cells by event time e = t - g, weighted by size.

    Negative e would be pre-treatment periods. They are a diagnostic, not a
    test: estimates near zero are consistent with parallel trends and do not
    establish it.
    """
    rows = []
    for e, group in cells.groupby("e"):
        rows.append({"e": e, "att": np.average(group["att"], weights=group["n"])})
    return pd.DataFrame(rows)

# your clustered-inference wrapper (Step 3.3)
def clustered_inference(model, panel, cluster="board"):
    """Return model results with standard errors clustered by a panel column."""
    if cluster not in panel.columns:
        raise ValueError(f"Cluster column {cluster!r} is not in the panel.")

    return model.get_robustcov_results(
        cov_type="cluster",
        groups=panel[cluster],
    )


# your bootstrap (Task 4.1)
def clean_att(panel):
    rows = []
    for g in sorted(panel.loc[panel["g"] < NEVER, "g"].unique()):
        cohort = panel[panel["g"] == g]
        baseline = g - 1
        for t in sorted(panel["t"].unique()):
            if t < g:
                continue
            controls = panel[panel["g"] > t]
            if controls.empty:
                continue

            treated_change = (
                cohort.loc[cohort["t"] == t, "outcome"].mean()
                - cohort.loc[cohort["t"] == baseline, "outcome"].mean()
            )
            control_change = (
                controls.loc[controls["t"] == t, "outcome"].mean()
                - controls.loc[controls["t"] == baseline, "outcome"].mean()
            )
            rows.append({
                "g": g,
                "t": t,
                "att": treated_change - control_change,
                "n": cohort.loc[cohort["t"] == t, "county"].nunique(),
            })

    cells = pd.DataFrame(rows)
    if cells.empty or cells["att"].isna().any():
        raise ValueError("No valid ATT cells.")
    return cells, np.average(cells["att"], weights=cells["n"])

def bootstrap_clean_att(panel, n_boot=999, seed=42, cluster="board"):
    if cluster not in panel.columns:
        raise ValueError(f"Cluster column {cluster!r} is not in the panel.")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")

    _, estimate = clean_att(panel)
    cluster_ids = panel[cluster].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    draws = []

    for _ in range(n_boot):
        sampled_ids = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        pieces = []
        for draw_id, original_id in enumerate(sampled_ids):
            piece = panel.loc[panel[cluster] == original_id].copy()
            piece[cluster] = draw_id
            piece["county"] = piece["county"].map(
                lambda county: f"{draw_id}_{county}"
            )
            pieces.append(piece)

        try:
            _, draw_estimate = clean_att(pd.concat(pieces, ignore_index=True))
            if np.isfinite(draw_estimate):
                draws.append(draw_estimate)
        except ValueError:
            continue

    if not draws:
        raise RuntimeError("No valid bootstrap samples; check the panel.")

    draws = np.asarray(draws)
    return {
        "estimate": estimate,
        "draws": draws,
        "ci_95": np.percentile(draws, [2.5, 97.5]),
        "n_successful": len(draws),
    }

def wild_cluster_bootstrap(
    model,
    panel,
    n_boot=999,
    seed=42,
    cluster="board",
    treatment="treated"
):

    if cluster not in panel.columns:
        raise ValueError(f"Cluster column {cluster!r} is not in the panel.")

    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")

    if treatment not in model.params.index:
        raise ValueError(f"{treatment!r} is not a model coefficient.")

    # Original estimate and standard error
    observed_estimate = model.params[treatment]

    observed_clustered = model.get_robustcov_results(
        cov_type="cluster",
        groups=panel[cluster]
    )

    treatment_idx = model.model.exog_names.index(treatment)
    observed_se = observed_clustered.bse[treatment_idx]
    observed_t = observed_estimate / observed_se

    # Fit the restricted model under H0: treatment effect = 0
    restricted_formula = model.model.formula
    restricted_formula = restricted_formula.replace(
        f"+ {treatment}", ""
    ).replace(
        f"{treatment} + ", ""
    )

    import statsmodels.formula.api as smf

    restricted_model = smf.ols(
        restricted_formula,
        data=panel
    ).fit()

    fitted_null = restricted_model.fittedvalues.to_numpy()
    residuals_null = restricted_model.resid.to_numpy()

    cluster_ids = panel[cluster].to_numpy()
    unique_clusters = np.unique(cluster_ids)

    rng = np.random.default_rng(seed)
    bootstrap_t = []

    for _ in range(n_boot):

        # One random weight per board
        weights = dict(zip(
            unique_clusters,
            rng.choice([-1, 1], size=len(unique_clusters))
        ))

        wild_weights = np.array([
            weights[c] for c in cluster_ids
        ])

        # Generate bootstrap outcome under the null
        bootstrap_panel = panel.copy()
        bootstrap_panel["_wild_y"] = (
            fitted_null + residuals_null * wild_weights
        )

        # Refit the original model
        bootstrap_model = smf.ols(
            model.model.formula.replace(
                model.model.endog_names, "_wild_y"
            ),
            data=bootstrap_panel
        ).fit()

        bootstrap_clustered = bootstrap_model.get_robustcov_results(
            cov_type="cluster",
            groups=bootstrap_panel[cluster]
        )

        boot_estimate = bootstrap_model.params[treatment]
        boot_se = bootstrap_clustered.bse[
            bootstrap_model.model.exog_names.index(treatment)
        ]

        if boot_se > 0:
            bootstrap_t.append(boot_estimate / boot_se)

    bootstrap_t = np.asarray(bootstrap_t)

    p_value = (
        1 + np.sum(np.abs(bootstrap_t) >= abs(observed_t))
    ) / (len(bootstrap_t) + 1)

    return {
        "estimate": observed_estimate,
        "t_statistic": observed_t,
        "p_value": p_value,
        "bootstrap_t": bootstrap_t,
        "n_successful": len(bootstrap_t)
    }

if __name__ == "__main__":
    import statsmodels.formula.api as smf

    # Check the two-by-two helper.
    example = pd.DataFrame({
        "treated": [0, 0, 1, 1],
        "post": [0, 1, 0, 1],
        "outcome": [1.0, 2.0, 3.0, 6.0],
    })
    assert np.isclose(two_by_two(example, "outcome", "treated", "post"), 2.0)

    # Check clustered inference 
    rows = []
    for cluster_id in range(8):
        treated = int(cluster_id >= 4)
        for post in (0, 1):
            outcome = 1 + 3 * treated + 2 * post + 2 * treated * post
            rows.append({
                "cluster": cluster_id,
                "treated": treated,
                "post": post,
                "outcome": outcome,
            })

    cluster_example = pd.DataFrame(rows)
    fit = smf.ols(
        "outcome ~ treated + post + treated:post",
        data=cluster_example,
    ).fit()
    clustered = clustered_inference(fit, cluster_example, cluster="cluster")
    assert clustered.cov_type == "cluster"

    # Check the bootstrap 
    rows = []
    for board in range(8):
        g = 2 if board < 4 else NEVER
        for county_in_board in range(2):
            county = board * 2 + county_in_board
            for t in (1, 2, 3):
                treated = int(g < NEVER and t >= g)
                rows.append({
                    "board": board,
                    "county": county,
                    "g": g,
                    "t": t,
                    "outcome": 10 + county + t + 2 * treated,
                })

    panel_example = pd.DataFrame(rows)
    boot = bootstrap_clean_att(
        panel_example, n_boot=99, seed=42, cluster="board"
    )
    assert np.isclose(boot["estimate"], 2.0)
    assert boot["n_successful"] > 0

    print("did_toolkit smoke tests passed.")
