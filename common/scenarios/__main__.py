import argparse
import json
from pathlib import Path

from common.initial_sequences.io import _invalid_constant, _unique_pairs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-background", required=True, type=Path,
                        help="directory produced by common.initial_sequences")
    parser.add_argument("--roads", required=True, type=Path,
                        help="the road snapshot used to construct the initial plan")
    parser.add_argument("--config", type=Path, help="scenario configuration JSON object")
    parser.add_argument("--seed", type=int, default=0, help="case seed; event seed offsets come from the config")
    parser.add_argument("--instance-id", help="explicit identity for the resulting recovery input")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="destination for recovery_input.json and the event summary")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("scenario output directory must be empty")

    from common.config import routing_options
    from common.data.roads.routing import Routing
    from common.data.roads.io import load_network
    from common.initial_sequences.workflow import load_initial_background
    from common.scenarios.parameters import normalize_scenario_config
    from common.scenarios.workflow import ScenarioConstructionError, build_scenario, save_scenario

    record = None
    if args.config is not None:
        try:
            record = json.loads(args.config.read_text(encoding="utf-8"),
                                parse_constant=_invalid_constant, object_pairs_hook=_unique_pairs)
        except (OSError, ValueError) as error:
            parser.error(f"scenario configuration: {error}")
    try:
        config = normalize_scenario_config(record)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    roads = load_network(args.roads)
    background = load_initial_background(args.initial_background, Routing(roads, **routing_options("formal")))
    try:
        scenario = build_scenario(background.replay, background.snapshot, config,
                                  seed=args.seed, instance_id=args.instance_id)
    except ScenarioConstructionError as error:
        import sys
        print(json.dumps(error.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True), file=sys.stderr)
        return 1
    save_scenario(scenario, args.output_dir)
    print(json.dumps(scenario.summary(), ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
