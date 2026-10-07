"""Create conservative within-grade repeat units without filling any input values."""
from pathlib import Path
import argparse
import pandas as pd

FEATURES=['C','Mn','Si','Cr','Ni','Mo','V','Cu','Al','austenitize_T','temper_T']

def build(raw_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite {output_path}; choose a new output path.')
    raw=pd.read_csv(raw_path)
    required=set(FEATURES+['heat_id','grade_id','tensile_strength'])
    if not required.issubset(raw.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(raw.columns))}')
    if raw.heat_id.isna().any() or raw.heat_id.duplicated().any() or raw.grade_id.isna().any():
        raise ValueError('Sample identifiers must be present and unique, and grade IDs present.')
    if 'analysis_unit' in raw:
        raise ValueError('Input already contains analysis_unit; supply the original open CSV.')
    signature=pd.util.hash_pandas_object(raw[FEATURES].astype(float),index=False)
    groups=pd.DataFrame({'grade':raw.grade_id,'signature':signature})
    size=groups.groupby(['grade','signature'])['signature'].transform('size')
    unit=raw.heat_id.copy()
    unit.loc[size>1]='repeat::'+raw.loc[size>1,'grade_id'].astype(str)+'::'+signature[size>1].astype(str)
    raw['analysis_unit']=unit
    output_path.parent.mkdir(parents=True,exist_ok=True)
    raw.to_csv(output_path,index=False)
    print(f'Wrote {len(raw)} rows and {raw.analysis_unit.nunique()} units to {output_path}.')
    print('No targets, predictors or labels were filled or changed. Cross-grade input matches are retained.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    build(args.raw,args.out)
