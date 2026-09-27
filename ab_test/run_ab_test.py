"""A/B test: static-threshold alerting (A) vs ML alerting (B), randomized by router.

PRE-REGISTRATION (fixed before looking at results; the constants below enforce it)
    Question     Does ML alerting cut false alerts without missing more incidents?
    Unit         Router. Randomizing individual alerts would put one router's alerts in
                 both arms and treat correlated alerts as independent samples.
    Assignment   50/50, stratified by role (core/aggregation/edge), fixed seed.
    Period       The two held-out test days (never used for training or model selection).
    Primary      False alerts per router-day. Defined for every router, including
                 routers with no alerts (a per-router false-alert *ratio* is not).
    Guardrail    Episode recall (share of real incidents that fired at least one alert).
                 Non-inferiority margin: B may not be more than 5 points worse than A.
    Test         OLS of per-router false alerts/day on arm + role fixed effects, robust
                 (HC3) standard errors, two-sided, alpha = 0.05: analyse the way you
                 randomized. Welch t-test and Mann-Whitney U reported as robustness checks.
                 (First version used Welch alone; the A/A check showed it was far too
                 conservative, 0.2% false-positive rate vs 5% nominal, because it ignores
                 the stratification. Kept in the output to show the difference.)
                 95% CIs by cluster bootstrap (resample routers, not alerts).
    Power        From the PRE-period (validation day, arm A behaviour): routers per arm
                 needed to detect a 30% reduction at 80% power.

Also shown, deliberately: the naive alert-level two-proportion z-test, and why it's wrong here.

    python ab_test/run_ab_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportions_ztest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from rules import rule_alerts  # noqa: E402
from train import load_features, split_by_time  # noqa: E402

ALPHA = 0.05
POWER = 0.80
TARGET_REDUCTION = 0.30
NONINFERIORITY_MARGIN = 0.05
SEED = 20260927
N_BOOT = 2000
N_AA = 500


def assign_arms(routers: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Stratified 50/50: shuffle within each role, alternate A/B. Keeps the arms balanced
    on the thing that most drives alert volume (role -> traffic -> latency profile)."""
    rng = np.random.default_rng(seed)
    arm = {}
    for _, g in routers.groupby("role"):
        ids = rng.permutation(g["router_id"].to_numpy())
        for i, r in enumerate(ids):
            arm[r] = "A" if i % 2 == 0 else "B"
    return routers["router_id"].map(arm)


def per_router(df: pd.DataFrame, alert_col: str) -> pd.DataFrame:
    days = df["date"].nunique()
    fa = (df[alert_col] & ~df["is_anomaly"]).groupby(df["router_id"]).sum()
    alerts = df[alert_col].groupby(df["router_id"]).sum()
    ep = df[df["episode_id"] >= 0].groupby(["router_id", "episode_id"])[alert_col].any()
    out = pd.DataFrame({"false_alerts_per_day": fa / days, "alerts": alerts, "false_alerts": fa})
    out["episodes"] = ep.groupby(level=0).size().reindex(out.index, fill_value=0)
    out["episodes_caught"] = ep.groupby(level=0).sum().reindex(out.index, fill_value=0)
    return out


def stratified_test(pr: pd.DataFrame):
    """Treatment effect of arm B with role fixed effects, heteroskedasticity-robust SEs."""
    fit = smf.ols("false_alerts_per_day ~ C(arm, Treatment('A')) + C(role)", data=pr).fit(cov_type="HC3")
    term = "C(arm, Treatment('A'))[T.B]"
    return fit.params[term], fit.pvalues[term], fit.conf_int().loc[term].tolist()


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, stat, rng, n=N_BOOT):
    """Cluster bootstrap: resample ROUTERS with replacement within each arm."""
    diffs = [stat(b.sample(len(b), replace=True, random_state=rng)) -
             stat(a.sample(len(a), replace=True, random_state=rng)) for _ in range(n)]
    return np.percentile(diffs, [2.5, 97.5])


def recall(g: pd.DataFrame) -> float:
    return g["episodes_caught"].sum() / max(g["episodes"].sum(), 1)


def false_alert_ratio(g: pd.DataFrame) -> float:
    return g["false_alerts"].sum() / max(g["alerts"].sum(), 1)


def main() -> None:
    rng = np.random.RandomState(SEED)
    df = load_features(ROOT / "data/features")
    _, pre, _ = split_by_time(df)
    pre = pre.copy()
    test = pd.read_parquet(ROOT / "artifacts/output/test_predictions.parquet")  # ML champion predictions

    # Both strategies scored on the same rows; each router only "experiences" its arm's alerts.
    _, test["alert_A"], _ = rule_alerts(test)
    test["alert_B"] = test["alert"]
    routers = test[["router_id", "role"]].drop_duplicates()
    routers["arm"] = assign_arms(routers).to_numpy()
    test = test.merge(routers[["router_id", "arm"]], on="router_id")
    test["served_alert"] = np.where(test["arm"] == "A", test["alert_A"], test["alert_B"])

    # ---- power analysis on the PRE period (arm A behaviour, before the experiment) -------
    _, pre["alert_A"], _ = rule_alerts(pre)
    pre_r = per_router(pre, "alert_A")["false_alerts_per_day"]
    effect = TARGET_REDUCTION * pre_r.mean() / pre_r.std()
    n_needed = TTestIndPower().solve_power(effect_size=effect, alpha=ALPHA, power=POWER)
    n_per_arm = len(routers) // 2
    mde = TTestIndPower().solve_power(nobs1=n_per_arm, alpha=ALPHA, power=POWER) * pre_r.std() / pre_r.mean()

    # ---- primary metric ---------------------------------------------------------------
    pr = per_router(test, "served_alert").join(routers.set_index("router_id")[["arm", "role"]])
    A, B = pr[pr["arm"] == "A"], pr[pr["arm"] == "B"]
    effect_b, strat_p, strat_ci = stratified_test(pr)
    welch = stats.ttest_ind(B["false_alerts_per_day"], A["false_alerts_per_day"], equal_var=False)
    mwu = stats.mannwhitneyu(B["false_alerts_per_day"], A["false_alerts_per_day"], alternative="two-sided")
    ci_fa = boot_diff(A, B, lambda g: g["false_alerts_per_day"].mean(), rng)

    # ---- guardrail: episode recall, non-inferiority --------------------------------------
    rec_a, rec_b = recall(A), recall(B)
    ci_rec = boot_diff(A, B, recall, rng)
    noninferior = ci_rec[0] > -NONINFERIORITY_MARGIN

    # ---- secondary: false-alert ratio (share of pages that were noise) --------------------
    far_a, far_b = false_alert_ratio(A), false_alert_ratio(B)
    ci_far = boot_diff(A, B, false_alert_ratio, rng)

    # ---- the WRONG test, for contrast: alerts treated as independent ------------------------
    naive = proportions_ztest([B["false_alerts"].sum(), A["false_alerts"].sum()],
                              [B["alerts"].sum(), A["alerts"].sum()])
    # Design effect: how much the within-router correlation inflates the variance.
    m = pr["alerts"].replace(0, np.nan).mean()
    fa_flag = test[test["served_alert"]].assign(fa=lambda d: ~d["is_anomaly"])
    icc = fa_flag.groupby("router_id")["fa"].mean().var() / max(fa_flag["fa"].var(), 1e-12)
    deff = 1 + (m - 1) * min(icc, 1.0)

    # ---- A/A validation: is the procedure's false-positive rate really ~5%? ----------------
    aa_router = per_router(test, "alert_A").join(routers.set_index("router_id")["role"])
    aa_rng = np.random.default_rng(SEED + 1)
    aa_welch, aa_strat = [], []
    for _ in range(N_AA):
        arm = assign_arms(aa_router.reset_index()[["router_id", "role"]], seed=int(aa_rng.integers(1e9)))
        arm.index = aa_router.index
        aa_welch.append(stats.ttest_ind(aa_router.loc[arm == "B", "false_alerts_per_day"],
                                        aa_router.loc[arm == "A", "false_alerts_per_day"], equal_var=False).pvalue)
        aa_strat.append(stratified_test(aa_router.assign(arm=arm))[1])
    aa_fpr_welch = float(np.mean(np.array(aa_welch) < ALPHA))
    aa_fpr = float(np.mean(np.array(aa_strat) < ALPHA))

    res = {
        "routers_per_arm": {"A": len(A), "B": len(B)},
        "power_routers_needed_per_arm": round(float(n_needed), 1),
        "mde_relative_at_n": round(float(mde), 3),
        "A_false_alerts_per_router_day": round(A["false_alerts_per_day"].mean(), 4),
        "B_false_alerts_per_router_day": round(B["false_alerts_per_day"].mean(), 4),
        "diff_false_alerts_per_router_day": round(B["false_alerts_per_day"].mean() - A["false_alerts_per_day"].mean(), 4),
        "diff_ci95": [round(x, 4) for x in ci_fa],
        "stratified_effect": round(float(effect_b), 4), "stratified_ci95": [round(x, 4) for x in strat_ci],
        "stratified_p": float(strat_p),
        "welch_t": round(float(welch.statistic), 3), "welch_p": float(welch.pvalue),
        "mannwhitney_p": float(mwu.pvalue),
        "A_episode_recall": round(rec_a, 4), "B_episode_recall": round(rec_b, 4),
        "recall_diff_ci95": [round(x, 4) for x in ci_rec], "noninferior": bool(noninferior),
        "A_false_alert_ratio": round(far_a, 4), "B_false_alert_ratio": round(far_b, 4),
        "far_diff_ci95": [round(x, 4) for x in ci_far],
        "naive_alert_level_z_p": float(naive[1]), "design_effect": round(float(deff), 2),
        "aa_false_positive_rate": aa_fpr, "aa_false_positive_rate_welch": aa_fpr_welch,
    }
    decision = ("SHIP B" if strat_p < ALPHA and res["diff_false_alerts_per_router_day"] < 0 and noninferior
                else "DO NOT SHIP")
    res["decision"] = decision

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("ab-test-alerting")
    with mlflow.start_run(run_name="rules-vs-ml-router-randomized"):
        mlflow.log_params({"alpha": ALPHA, "power": POWER, "target_reduction": TARGET_REDUCTION,
                           "unit": "router", "stratified_by": "role", "seed": SEED})
        mlflow.log_metrics({k: v for k, v in res.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        mlflow.log_dict(res, "ab_result.json")

    out = ROOT / "artifacts" / "ab_test_result.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
