import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data.roads.adapters.osm import OSMBuildConfig, OSMRoadAdapter
from common.data.roads.io import save_network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--target-crs', required=True)
    parser.add_argument('--bounds', nargs=4, type=float,
                        metavar=('WEST','SOUTH','EAST','NORTH'))
    parser.add_argument('--name')
    parser.add_argument('--version', default='1')
    parser.add_argument('--travel-mode', choices=('cycling','driving'), default='cycling')
    parser.add_argument('--speed-m-per-s', type=float, default=4.)
    args = parser.parse_args()
    try:
        roads = OSMRoadAdapter().adapt(args.source, config=OSMBuildConfig(
            name=args.name or args.source.stem, version=args.version,
            target_crs=args.target_crs, travel_mode=args.travel_mode,
            speed_m_per_s=args.speed_m_per_s,
            bounds=None if args.bounds is None else tuple(args.bounds)))
    except ModuleNotFoundError as error:
        if error.name != 'osmium':
            raise
        parser.error('OSM import requires osmium; install requirements/osm.txt')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_network(roads, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
