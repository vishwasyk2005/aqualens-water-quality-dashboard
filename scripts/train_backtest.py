"""Train the separate pre-2022 DET historical-validation checkpoint."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aqualens.data import load_data
from aqualens.backtest import train_historical_backtest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/modis_observations.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'models/historical_validation')
    parser.add_argument('--epochs',type=int,default=160)
    args=parser.parse_args()
    frame,_=load_data(args.data)
    summary=train_historical_backtest(frame,args.output,epochs=args.epochs)
    print('Historical validation saved:',args.output)
    print('Selected candidate',summary['selected_candidate'],'epoch',summary['best_epoch'])
    for row in summary['metrics']:print(row)
if __name__=='__main__':main()
