```
[Prep] Act as an applied microeconometrician who has implemented
cluster-robust inference for panels with very few clusters.

[Request] Implement the wild cluster bootstrap exactly to the
specification below, for a county-year panel with treatment assigned
at the level of eight regional boards. Impose the null, apply
Rademacher weights at the cluster level, enumerate all 256 weight
vectors rather than resampling, and return the bootstrap p-value for
the treatment coefficient.
Fit the restricted model: your DiD regression with the treatment coefficient forced to zero. Keep the restricted residuals ũi and fitted values ỹi.
Draw one Rademacher weight per board — +1 or −1, each with probability one half. One weight per board, not per observation.
Build a synthetic outcome y*i = ỹi + wb(i) ũi, where b(i) is county i’s board.
Re-estimate the unrestricted model on y* and store the cluster-robust t-statistic on the treatment coefficient.
The bootstrap p-value is the share of | t* | at least as large as your actual | t |.
With 8 boards there are only 28 = 256 distinct weight vectors, so enumerate all 256 rather than resampling. Say in a text cell why that makes the number of replications irrelevant here, and what it implies for the smallest p-value you are able to report.

[Iterate] numpy, pandas and statsmodels only. Reuse my panel's column
names. The function must accept the cluster variable as an argument.

[Mechanism Check] Explain why the weights are applied to residuals at
the CLUSTER level rather than the observation level, why imposing the
null matters, and what changes when the number of clusters drops
below roughly ten.

[Evaluate] Run it on my panel and compare the p-value against the
naive cluster-robust one. Tell me which conclusion changes.
```Panel DataFrame: panel

Columns:
- county: county identifier
- board: regional board identifier (8 boards)
- t: year/time period
- g: treatment adoption year (10**9 means never treated)
- outcome: dependent variable
- treated: binary treatment indicator (1 after adoption, 0 otherwise) TWFE model: twfe

Model formula:
outcome ~ treated + C(county) + C(t)

Treatment coefficient: treated
Cluster variable: board
Number of clusters: 8 

I can't run this on your panel because no data or file came through in this conversation, only the schema. Below is the function and the explanation. The last cell is the call that produces the comparison, and it ends with how to read the result.

## Code

```python
import itertools
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

FORMULA = "outcome ~ treated + C(county) + C(t)"

def wild_cluster_bootstrap(df, cluster="board", formula=FORMULA, coef="treated"):
    """Exact (fully enumerated) wild cluster bootstrap-t, null imposed,
    Rademacher weights at the cluster level."""
    full = smf.ols(formula, data=df)
    y, X, names = full.endog, full.exog, full.exog_names
    j = names.index(coef)

    # Cluster codes aligned to the rows actually used (patsy drops NaNs)
    groups = pd.factorize(df.loc[full.data.row_labels, cluster])[0]
    G = groups.max() + 1
    n, k = X.shape
    rank = np.linalg.matrix_rank(X)

    # CR1 small-sample factor (matches statsmodels' default cluster correction)
    c = G / (G - 1) * (n - 1) / (n - rank)
    XtXinv = np.linalg.pinv(X.T @ X)
    r = X @ XtXinv[:, j]          # beta_j = r'y

    def cluster_t(yv):
        beta = XtXinv @ (X.T @ yv)
        u = yv - X @ beta
        s = np.bincount(groups, weights=r * u, minlength=G)  # cluster scores for coef j
        return beta[j] / np.sqrt(c * np.sum(s ** 2))

    # Actual t, cross-checked against statsmodels
    res = full.fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)
    t_obs = cluster_t(y)
    assert np.isclose(t_obs, res.tvalues[j]), "t mismatch vs statsmodels"

    # Restricted model: treatment coefficient forced to zero
    Xr = np.delete(X, j, axis=1)
    b_r = np.linalg.lstsq(Xr, y, rcond=None)[0]
    y_tilde = Xr @ b_r
    u_tilde = y - y_tilde

    # Enumerate all 2^G Rademacher vectors, one weight per cluster
    W = np.array(list(itertools.product([-1.0, 1.0], repeat=G)))
    t_star = np.array([cluster_t(y_tilde + w[groups] * u_tilde) for w in W])

    p_boot = np.mean(np.abs(t_star) >= np.abs(t_obs) - 1e-10)
    return {
        "coef": res.params[j], "t_obs": t_obs,
        "p_naive_t_G-1": res.pvalues[j],
        "p_naive_normal": 2 * (1 - __import__("math").erf(abs(t_obs) / 2 ** 0.5) / 2 - 0.5) if False else None,
        "p_boot": p_boot, "n_draws": len(W), "t_star": t_star,
    }
```

The `p_naive_normal` line is clutter, so use this cleaner version in its place (no scipy needed):

```python
from math import erfc, sqrt
# inside the return dict:
"p_naive_normal": erfc(abs(t_obs) / sqrt(2)),
```

```python
# Run on your panel
out = wild_cluster_bootstrap(panel, cluster="board")
print({k: v for k, v in out.items() if k != "t_star"})
```

## Text cell: why the number of replications is irrelevant

With 8 boards and weights in {−1, +1}, there are exactly 2⁸ = 256 distinct weight vectors. Enumerating all of them gives the exact bootstrap distribution, with no Monte Carlo error. Drawing B random vectors would only re-sample from these same 256 points, with repeats. So B is not a tuning parameter, and the p-value is a deterministic function of the data.

The grid is coarse in two ways:

- **Floor at 1/256.** The all-(+1) vector gives y* = ỹ + ũ = y, which reproduces the actual t. So the actual |t| is always in the reference set, and p ≥ 1/256 ≈ 0.0039.
- **Floor at 2/256.** ỹ lies in the restricted column space, so β̂_treated on y* is linear in the weighted residuals, and −w gives exactly −t*. The |t*| values therefore come in ± pairs (128 distinct values), and the all-(−1) vector always ties the actual |t|. The smallest reportable two-sided p-value is **2/256 = 1/128 ≈ 0.0078**. If your naive p is 0.001, the bootstrap cannot show anything below 0.0078.

## Mechanism check

**Why weights go on residuals at the cluster level.**
- Errors within a board are arbitrarily correlated and heteroskedastic, and cluster-robust inference only assumes independence across boards.
- A single weight w_b multiplies all of board b's residuals, so the within-board correlation and variance pattern is preserved exactly. The weights only randomize the sign of each board's whole residual vector.
- Observation-level weights would break that dependence and yield an iid-like error structure. The bootstrap would then understate the variance of the treatment coefficient, which is the same failure as ignoring clustering.

**Why impose the null.**
- The bootstrap p-value asks how often a t this extreme arises when β_treated = 0 is true. That requires the data-generating process to satisfy the null.
- Resampling unrestricted residuals instead centers the distribution on the estimated effect, so you would have to recenter t* around β̂. That version has worse size in small samples.
- With the null imposed, y* has true coefficient zero by construction, and t* is a valid draw from the null distribution of the t-statistic. Using a studentized statistic (bootstrap-t) gives asymptotic refinement over bootstrapping the coefficient alone.

**What changes below roughly ten clusters.**
- The cluster-robust variance estimator is biased downward, and the t(G−1) approximation is poor. The naive test over-rejects, sometimes badly, especially with few treated boards or unbalanced cluster sizes.
- The wild cluster bootstrap-t (Cameron, Gelbach and Miller 2008) corrects much of this, because it approximates the actual finite-sample distribution of the t-statistic rather than relying on the asymptotic one.
- Rademacher weights become discrete: 2^G points, and in effect 2^(G−1) distinct |t*|. At G = 8 that is the 1/128 floor above. Below about G = 10 the literature suggests Webb's six-point weights (MacKinnon and Webb 2017), which give 6^G draws.
- If only one or two boards ever adopt treatment, the bootstrap can fail in the other direction and under-reject. Check how many boards contain treated counties (`panel.groupby("board")["treated"].max().sum()`).

## Evaluate

I can't give you the p-value comparison without your data. Here is how to read the output once you run it:

- **`p_naive_t_G-1`** is the usual cluster-robust p-value with 7 degrees of freedom. `p_naive_normal` is what you get if your software uses the normal approximation, which is more anti-conservative.
- **`p_boot`** is the exact bootstrap p-value.
- **Conclusion changes** if the two fall on opposite sides of your threshold, typically naive p < 0.05 and `p_boot` ≥ 0.05. That is the common pattern with 8 clusters.
- If `p_boot` is near 0.0078, the result is at the resolution limit of the design.

If you paste the output dict or a sample of the panel, I'll interpret it. Also, since adoption is staggered (`g` varies), the TWFE coefficient itself may be contaminated by heterogeneous effects. The bootstrap fixes the inference, not that issue.

```
[Prep] Act as an applied microeconometrician who has implemented
cluster-robust inference for panels with very few clusters.

[Request] Implement the wild cluster bootstrap exactly to the
specification below, for a county-year panel with treatment assigned
at the level of eight regional boards. Impose the null, apply
Rademacher weights at the cluster level, enumerate all 256 weight
vectors rather than resampling, and return the bootstrap p-value for
the treatment coefficient.
Fit the restricted model: your DiD regression with the treatment coefficient forced to zero. Keep the restricted residuals ũi and fitted values ỹi.
Draw one Rademacher weight per board — +1 or −1, each with probability one half. One weight per board, not per observation.
Build a synthetic outcome y*i = ỹi + wb(i) ũi, where b(i) is county i’s board.
Re-estimate the unrestricted model on y* and store the cluster-robust t-statistic on the treatment coefficient.
The bootstrap p-value is the share of | t* | at least as large as your actual | t |.
With 8 boards there are only 28 = 256 distinct weight vectors, so enumerate all 256 rather than resampling. Say in a text cell why that makes the number of replications irrelevant here, and what it implies for the smallest p-value you are able to report.

[Iterate] numpy, pandas and statsmodels only. Reuse my panel's column
names. The function must accept the cluster variable as an argument.

[Mechanism Check] Explain why the weights are applied to residuals at
the CLUSTER level rather than the observation level, why imposing the
null matters, and what changes when the number of clusters drops
below roughly ten.

[Evaluate] Run it on my panel and compare the p-value against the
naive cluster-robust one. Tell me which conclusion changes.
```Panel DataFrame: panel

Columns:
- county: county identifier
- board: regional board identifier (8 boards)
- t: year/time period
- g: treatment adoption year (10**9 means never treated)
- outcome: dependent variable
- treated: binary treatment indicator (1 after adoption, 0 otherwise) TWFE model: twfe

```
The fitted model `twfe` uses exactly the same rows and order as `panel`.
```

Model formula:
outcome ~ treated + C(county) + C(t)

Treatment coefficient: treated
Cluster variable: board
Number of clusters: 8 


I can't give you the p-values yet. No data came through in this chat, only the description of `panel` and `twfe`, so I haven't run anything and won't invent numbers. Below is the implementation, followed by the explanations you asked for and a way to read the output once you run it.

## Code cell

```python
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def _cluster_t(y, X, groups, k):
    """Cluster-robust (CR1, statsmodels default) t-stat for exog column k."""
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return res.params[k] / res.bse[k], res


def wild_cluster_bootstrap_enum(
    df,
    cluster,                                   # name of the cluster column, e.g. "board"
    formula="outcome ~ treated + C(county) + C(t)",
    treat="treated",
    tol=1e-10,
):
    """
    Wild cluster bootstrap-t, null imposed, Rademacher weights,
    full enumeration of the 2**G weight vectors.
    """
    # --- unrestricted model (fixed design; only y changes in the bootstrap)
    unr = smf.ols(formula, data=df).fit()
    X = unr.model.exog
    y = unr.model.endog
    k = unr.model.exog_names.index(treat)

    # cluster codes aligned with the rows the model actually used
    used = df.loc[unr.model.data.row_labels, cluster]
    codes, uniques = pd.factorize(used)
    G = len(uniques)

    t_actual, res_actual = _cluster_t(y, X, codes, k)

    # --- restricted model: same formula with the treatment term removed (beta = 0)
    rhs = [s.strip() for s in formula.split("~")[1].split("+")]
    restricted_formula = formula.split("~")[0].strip() + " ~ " + " + ".join(
        s for s in rhs if s != treat
    )
    rst = smf.ols(restricted_formula, data=df).fit()
    assert len(rst.resid) == len(y), "restricted/unrestricted rows differ"
    y_tilde = np.asarray(rst.fittedvalues)     # fitted values  (ỹ)
    u_tilde = np.asarray(rst.resid)            # residuals      (ũ)

    # --- enumerate all 2**G sign vectors, one weight per cluster
    t_star = np.empty(2 ** G)
    for j, w in enumerate(itertools.product([-1.0, 1.0], repeat=G)):
        w = np.asarray(w)
        y_star = y_tilde + w[codes] * u_tilde  # w_{b(i)} * ũ_i
        t_star[j], _ = _cluster_t(y_star, X, codes, k)

    p_boot = np.mean(np.abs(t_star) >= abs(t_actual) - tol)

    # --- conventional cluster-robust p-values for comparison
    p_naive_normal = float(res_actual.pvalues[k])           # statsmodels default: N(0,1)
    res_t = sm.OLS(y, X).fit(cov_type="cluster",
                             cov_kwds={"groups": codes}, use_t=True)
    p_naive_t = float(res_t.pvalues[k])                     # t(G-1)

    return {
        "coef": float(res_actual.params[k]),
        "t_actual": float(t_actual),
        "G": G,
        "n_weight_vectors": 2 ** G,
        "p_boot": float(p_boot),
        "p_naive_normal": p_naive_normal,
        "p_naive_t_G-1": p_naive_t,
        "t_star": t_star,
    }


out = wild_cluster_bootstrap_enum(panel, cluster="board")
print({k: v for k, v in out.items() if k != "t_star"})
```

## Text cell: why the number of replications is irrelevant, and the smallest p-value

With 8 boards and weights in {−1, +1}, there are exactly 2⁸ = 256 distinct weight vectors. Enumerating all of them gives the exact bootstrap distribution, so there is no Monte Carlo error and no replication count to choose. Drawing more random vectors would only repeat some of the same 256 and add noise.

Two things follow for the smallest reportable p-value:

- The all-(+1) vector gives y* = ỹ + ũ = y, which reproduces your actual t, so it always satisfies |t*| ≥ |t|.
- Flipping every sign (w → −w) flips the sign of t*, so the all-(−1) vector also gives |t*| = |t|.

So the 256 draws contain only 128 distinct values of |t*|, each appearing twice, and the p-value moves in steps of 2/256 = 1/128. The smallest p-value you can report is **2/256 ≈ 0.0078**. A result like "p = 0.000" is impossible here, and conventional 1% significance is only just reachable.

## Mechanism check

**Why weights go on residuals at the cluster level.** Errors within a board are arbitrarily correlated (serially and across counties), and the cluster-robust variance estimator allows for this. Observation-level weights would break that dependence structure, because each ũᵢ would be flipped independently and the bootstrap errors would look i.i.d., so the bootstrap would understate the variance. One weight per board flips all of a board's residuals together. The residuals keep their within-board covariance structure, and the weights only randomize the sign of each cluster's contribution. Because E[w] = 0 and E[w²] = 1, the cluster-level moments are preserved.

**Why impose the null.** The bootstrap p-value asks how often a t-statistic this large occurs when the true effect is zero. Building y* from the restricted fit (β = 0) makes the data-generating process satisfy H₀, so the t* distribution is a null distribution. If you resampled from the unrestricted fit, the bootstrap world would have the estimated effect built in, and you would need to recenter the test. In practice, imposing the null gives much better size control with few clusters.

**What changes below about ten clusters.**
- The cluster-robust variance estimator (CRVE) is biased downward, because it effectively has only G − 1 degrees of freedom of information about the error structure.
- The t-statistic is not well approximated by N(0,1), or even by t(G − 1) when cluster sizes or treatment exposure are unbalanced. Naive tests over-reject, often badly.
- The wild cluster bootstrap-t with the null imposed corrects much of this (Cameron, Gelbach and Miller 2008; MacKinnon and Webb 2017).
- Rademacher weights become discrete, which gives the 256-point support and the 1/128 grid above. Webb's six-point weights (6⁸ ≈ 1.7 million vectors) are the usual alternative when G < 12. The bootstrap also still degrades if only one or two boards are treated.

## Evaluate: how to read the output

Run the code cell and compare `p_boot` with `p_naive_normal` and `p_naive_t_G-1`:

- Naive p < 0.05 and `p_boot` ≥ 0.05: the finding is not robust to few-cluster inference. This is the usual direction of change.
- Both below 0.05: the conclusion survives, and the bootstrap p will typically be larger.
- Both above 0.05: no change in the conclusion.
- `p_boot` is expressed in units of 1/128, so a value like 0.0625 means 8 of the 128 distinct |t*| values met or exceeded your |t|.

If you paste the printed dictionary, or upload the panel, I'll interpret the result and tell you exactly which conclusion changes. 

```
[Prep] Act as the state agency's lead econometrician. You believe the
programme works and you are reading an external review that says your
TWFE estimate is uninformative.

[Request] Here is the reviewer's paragraph on parallel trends and
identification: <The flat pretrend near zero suggests that the treated and untreated groups were following similar trends before the treatment. This makes it more plausible that both groups would have continued along similar trends without the policy, but it does not prove it. There are also no uncertainty bars, so it is hard to tell how precise the estimates are or whether the differences could just be noise. For this design to hold, we would have to assume that without the policy, treated counties would have continued changing in the same way as untreated counties and that counties did not change their behavior ahead of the policy..>. Give the three strongest
objections you would raise in a written response.

[Evaluate] Rank them by how much damage each does to the reviewer's
conclusion.
```


# Response to the external review: identification

## Objection 1 (most damaging): the conclusion does not follow from the premises

The reviewer concedes that the flat pretrend makes parallel trends "more plausible," then says it "does not prove it." That is true of every difference-in-differences design, and of every observational design. The identifying assumption is untestable by construction. If "does not prove it" were enough to make an estimate uninformative, the whole quasi-experimental literature would be uninformative.

The right standard is whether the assumption is credible and whether the conclusion survives plausible violations. The review does the first half (it grants plausibility) and skips the second. It offers no specific confounder, no county-level shock that hit treated counties differentially, and no mechanism for diverging counterfactual trends. A critique that names no violation is a statement about epistemology, not about this estimate.

We will answer with a sensitivity analysis rather than a defence. We will report how large a post-treatment divergence in trends would have to be, relative to the pre-period deviations we observe, to overturn the result (Rambachan and Roth-style bounds). The reviewer can then judge whether those violations are credible.

This objection does the most damage because it attacks the leap from "not proven" to "uninformative," which is the review's actual conclusion. If it holds, the other two points can only support a weaker verdict like "suggestive but imprecise."

## Objection 2: anticipation is testable, and the event study already tests it

The reviewer's second assumption, that counties did not change behaviour ahead of the policy, is not a separate leap of faith. Anticipation shows up as lead coefficients that move away from zero in the periods just before implementation. Our leads are flat through the final pre-period, which is exactly the evidence against anticipation.

The institutional record helps too. We can document the gap between announcement and implementation, and whether counties had the administrative lead time or funding certainty to adjust early. If the policy was announced shortly before it took effect, anticipation on any meaningful scale is implausible. We can also test robustness by dropping the last pre-period or redefining treatment timing a period earlier.

This ranks second because it removes half of the reviewer's stated assumptions using evidence already in the figure. It is less damaging than Objection 1 because it addresses only one of the two assumptions and does not touch the broader logic.

## Objection 3: missing uncertainty bars is a presentation defect, not an identification defect

The absence of confidence intervals on the plot says nothing about whether the design is valid. It says the figure was incomplete. We will supply the intervals, with standard errors clustered at the level of treatment assignment, and report the joint test of the leads.

The reviewer's own reading is also internally inconsistent. If the pretrend estimates are noisy, that is an argument about precision, and precision is something we can quantify. Tight intervals around zero would turn the reviewer's "could just be noise" from a worry into a rebuttal.

This ranks last because it is the reviewer's most legitimate point, and the fix is to show the intervals. It does little damage to the "uninformative" conclusion unless the intervals turn out to be wide.

## Where we are exposed

Two points deserve candour before this goes out:

- **Objection 3 cuts both ways.** If the confidence intervals are wide enough to include meaningful pre-trends, the reviewer wins that point, and a flat point estimate cannot save us. Pretrend tests also have low power, so "we failed to reject" is weaker than it sounds. I would run the intervals and the sensitivity bounds before sending any rebuttal.
- **The reviewer did not raise the TWFE-specific problem.** If adoption was staggered, TWFE can put negative weights on some treated units, and a flat event study from a TWFE specification can mask that. A heterogeneity-robust estimator (Callaway–Sant'Anna, Sun–Abraham, or similar) as a robustness check would pre-empt the strongest version of the review's critique, and I would include it in our response. 