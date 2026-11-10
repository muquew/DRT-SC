import argparse
from pathlib import Path
from solvers.deco.data import prepare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepared=prepare(args.config,args.output)
    print('Prepared',len(prepared['episodes']),'DECO episode references')


if __name__=='__main__':
    main()
