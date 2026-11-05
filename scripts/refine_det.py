"""Run the bounded DET refinement on pre-2022 readings only."""
import argparse
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aqualens.data import load_data
from aqualens.det_refinement import train_refinement,VALIDATION_END

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/modis_observations.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'models/refinement_v3')
    parser.add_argument('--epochs',type=int,default=200)
    args=parser.parse_args()
    # Ingestion preserves the study area's fixed zone identifiers. Cut the
    # frame before the search: no 2022+ values enter its panel or objectives.
    frame,_=load_data(args.data)
    past=frame[pd.to_datetime(frame.date)<=pd.Timestamp(VALIDATION_END)].copy()
    del frame
    summary=train_refinement(past,args.output,epochs=args.epochs,
        original_checkpoint=ROOT/'models/historical_validation/det_historical.pt')
    print('Locked candidate:',summary['selected_candidate'],'epoch:',summary['best_epoch'])
    print('Validation normalized RMSE:',summary['validation_score'])
    print('Original DET validation reference:',summary['references'].get('original_frozen_det',{}).get('score'))
    print('Last-observed diagnostic:',summary['references']['last_observed_diagnostic']['score'])
    print('2022+ test data has NOT been scored by this search.')
if __name__=='__main__':main()
