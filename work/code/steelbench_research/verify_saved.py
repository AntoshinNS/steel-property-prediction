"""Recheck saved end-to-end results without refitting models."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument("run_directory")
args=parser.parse_args()
r=Path(args.run_directory)
assert json.loads((r/"run_status.json").read_text())["state"]=="complete"
p=pd.read_csv(r/"predictions.csv");m=pd.read_csv(r/"split_manifest.csv")
assert not p.duplicated(["protocol","sample"]).any()
for key,z in m.groupby(["protocol","outer_fold","inner_fold"]):
    train=set(z[z.role.str.endswith("train")]["sample"])
    test=set(z[z.role.str.endswith("test")|z.role.str.endswith("validation")]["sample"])
    assert not train & test, key
cur=pd.read_csv(r/"risk_retention.csv")
for key,z in cur[cur.requested_retention==1].groupby(["protocol","family"]):
    assert np.ptp(z.mae)<1e-10,key
for folder in r.glob("*_[0-9][0-9][0-9]"):
    v=np.load(folder/"attributions.npz")
    np.testing.assert_allclose(v["phi"].sum(2)+v["expected"][:,None],v["predictions"],atol=1e-6)
    saved=pd.read_csv(folder/"predictions.csv")
    np.testing.assert_allclose(v["predictions"].mean(0),saved.prediction)
print("Saved split boundaries, ensemble means, SHAP additivity, and full-retention equality: PASS")
