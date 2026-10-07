"""Meaningful arithmetic and information-boundary tests; stdlib unittest."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import (DEFAULT, permutation_attributions, partition, risk_curve,
                        load_data, ensemble_diagnostics)


class ProtocolTests(unittest.TestCase):
    def test_linear_shap_has_known_solution(self):
        x=np.array([[2.,3.,4.],[-1.,1.,0.]])
        bg=np.array([[0.,1.,2.],[1.,0.,0.]])
        w=np.array([3.,-2.,.5])
        f=lambda z:7+z@w
        phi,e=permutation_attributions(f,x,bg,pairs=1,seed=7)
        np.testing.assert_allclose(phi,(x-bg.mean(0))*w,atol=1e-12)
        self.assertAlmostEqual(e,f(bg).mean())

    def test_pairwise_interaction_antithetic_value(self):
        x=np.array([[2.,3.]])
        bg=np.array([[0.,0.],[1.,2.]])
        f=lambda z:z[:,0]*z[:,1]
        phi,e=permutation_attributions(f,x,bg,pairs=1,seed=9)
        # v(empty)=1, v({0})=2, v({1})=1.5, v(all)=6
        np.testing.assert_allclose(phi,[[2.75,2.25]],atol=1e-12)
        np.testing.assert_allclose(phi.sum(1)+e,f(x))

    def test_known_retention_arithmetic(self):
        a,rows=risk_curve([0,0,0,0],[1,2,3,4],[0,1,2,3],["a","b","c","d"],[.5,1.])
        self.assertEqual(rows[0]["mae"],1.5)
        self.assertEqual(rows[1]["mae"],2.5)
        self.assertEqual(a,2.)

    def test_ties_do_not_use_targets_or_row_order(self):
        args=([0,0,0,0],[4,3,2,1],[0,0,0,0],["d","c","b","a"])
        a,rr=risk_curve(*args,[.5,1.])
        ix=[3,0,2,1]
        b,ss=risk_curve(*[np.array(v)[ix] for v in args],[.5,1.])
        self.assertEqual(a,b);self.assertEqual(rr,ss)

    def data(self):
        rng=np.random.default_rng(12)
        d=pd.DataFrame({"sample":[f"s{i}" for i in range(24)],
          "grade":[f"g{i//2}" for i in range(24)],"family":[f"f{i//6}" for i in range(24)],
          "source":[f"s{i%3}" for i in range(24)],"unit":[f"h{i}" for i in range(24)],
          "x1":rng.normal(size=24),"x2":rng.normal(size=24),"y":rng.normal(600,30,24)})
        return d

    def test_all_group_splits(self):
        d=self.data()
        for protocol,key in (("family","family"),("grade","grade"),("random","unit"),("source","source")):
            seen=[]
            for a,b in partition(d,protocol,3,4):
                self.assertFalse(set(d.iloc[a][key])&set(d.iloc[b][key]));seen.extend(b)
            self.assertEqual(sorted(seen),list(range(len(d))))

    def test_source_unit_leak_fails(self):
        d=self.data();d.loc[1,"unit"]=d.loc[0,"unit"]
        with self.assertRaisesRegex(ValueError,"Leakage"):
            list(partition(d,"source",3,4))

    def test_target_not_used_in_outer_diagnostics(self):
        d=self.data();c=copy.deepcopy(DEFAULT)
        c.update(features=["x1","x2"],bootstrap_models=2,n_estimators=5,background_size=2,permutation_pairs=1)
        a=d.iloc[:18].copy();b=d.iloc[18:].copy()
        r1=ensemble_diagnostics(a,b,c,4)
        b.y=1e12
        r2=ensemble_diagnostics(a,b,c,4)
        pd.testing.assert_frame_equal(r1,r2)

    def test_missing_and_duplicate_schema_failures(self):
        c=copy.deepcopy(DEFAULT)
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"x.csv";pd.DataFrame({"wrong":[1,2]}).to_csv(f,index=False)
            with self.assertRaisesRegex(ValueError,"Missing columns"):
                load_data(f,c)
            d=self.data().rename(columns={"sample":"sample_id","unit":"heat_id","y":"tensile_strength"})
            c["features"]=["x1","x2"]
            d.loc[1,["x1","x2"]]=d.loc[0,["x1","x2"]]
            d.to_csv(f,index=False)
            with self.assertRaisesRegex(ValueError,"duplicates"):
                load_data(f,c)


if __name__=="__main__":unittest.main(verbosity=2)
