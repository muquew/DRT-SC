import argparse
import json
from pathlib import Path
from solvers.deco.training import train


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--device',choices=('cpu','auto','gpu'))
    args=parser.parse_args()
    print(json.dumps(train(args.config,args.output,resume=args.resume,device=args.device)))


if __name__=='__main__':
    main()
