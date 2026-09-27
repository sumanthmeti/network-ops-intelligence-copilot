import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ab_test"))
from run_ab_test import assign_arms  # noqa: E402


def routers():
    roles = ["core"] * 30 + ["aggregation"] * 70 + ["edge"] * 100
    return pd.DataFrame({"router_id": [f"R{i:04d}" for i in range(200)], "role": roles})


def test_assignment_is_balanced_within_every_role():
    r = routers().assign(arm=lambda d: assign_arms(d).to_numpy())
    counts = r.groupby(["role", "arm"]).size().unstack()
    assert (counts["A"] == counts["B"]).all()


def test_assignment_is_deterministic_per_seed_and_varies_across_seeds():
    r = routers()
    assert assign_arms(r, seed=1).equals(assign_arms(r, seed=1))
    assert not assign_arms(r, seed=1).equals(assign_arms(r, seed=2))
