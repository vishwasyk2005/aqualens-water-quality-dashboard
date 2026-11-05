"""Train DET only on actual uploaded historical observations."""
import argparse
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aqualens.forecast import train_models
from aqualens.data import load_data

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data'/'modis_observations.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'models')
    parser.add_argument('--epochs',type=int,default=160)
    args=parser.parse_args()
    frame,report=load_data(args.data)
    summary=train_models(frame,args.output,epochs=args.epochs)
    print('Saved real-data DET:',args.output/'det_v2.pt')
    print(pd.DataFrame(summary['metrics']).to_string(index=False))
    print('Selected configuration:',summary['config'],'epoch',summary['best_epoch'])
    print('Held-out scores use train-only checkpoint; production model is an explicit all-history refit.')
if __name__=='__main__': main()
