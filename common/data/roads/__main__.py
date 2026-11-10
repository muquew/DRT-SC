import argparse
from pathlib import Path

from common.data.roads.adapters.grid import GridBuildConfig, GridRoadBuilder
from common.data.roads.io import save_network


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--bounds', nargs=4, type=float, required=True,
                        metavar=('XMIN','YMIN','XMAX','YMAX'))
    parser.add_argument('--spacing-m', type=float, default=200.)
    parser.add_argument('--speed-m-per-s', type=float, default=4.)
    parser.add_argument('--crs', default='local:experimental-metres')
    parser.add_argument('--name', default='experimental-grid')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    roads = GridRoadBuilder().build(config=GridBuildConfig(name=args.name,
        version='1', crs=args.crs, bounds=tuple(args.bounds), spacing_m=args.spacing_m,
        speed_m_per_s=args.speed_m_per_s))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_network(roads, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
