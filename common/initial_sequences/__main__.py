import argparse
import json
from pathlib import Path

from common.initial_sequences.io import _invalid_constant, _unique_pairs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--planning-input", required=True, type=Path)
    parser.add_argument("--roads", required=True, type=Path)
    methods = parser.add_mutually_exclusive_group()
    methods.add_argument("--method", help="greedydp, foodmatch or ocd (default: greedydp)")
    methods.add_argument("--method-config", type=Path, help="JSON concise recipe or complete GeneratorConfig")
    parser.add_argument("--seed", type=int, default=0, help="default method seed; explicit config seed takes precedence")
    parser.add_argument("--parameters", help="JSON object of adapter parameters; used with --method")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--time-s", type=float)
    target.add_argument("--remaining-tasks", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.method_config is not None and args.parameters is not None:
        parser.error("--parameters belongs to --method; configure parameters inside --method-config")
    from common.config import routing_options
    from common.data.roads.routing import Routing
    from common.data.roads.io import load_network
    from common.initial_sequences.io import load_planning_input
    from common.initial_sequences.workflow import (
        INITIAL_BACKGROUND_FILES, build_initial_background, save_initial_background,
    )
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(
            (args.output_dir / name).exists() for name in INITIAL_BACKGROUND_FILES)):
        parser.error("initial background output must be empty of initial-plan files")

    def read_json(value):
        return json.loads(value, parse_constant=_invalid_constant, object_pairs_hook=_unique_pairs)
    if args.method_config is not None:
        method = read_json(args.method_config.read_text())
    else:
        method = {"name": args.method or "greedydp", "seed": args.seed,
                  "parameters": read_json(args.parameters) if args.parameters is not None else {}}
    roads = load_network(args.roads)
    planning_input = load_planning_input(args.planning_input, roads)
    background = build_initial_background(planning_input, Routing(roads, **routing_options("formal")), method,
        default_seed=args.seed, time_s=args.time_s, remaining_tasks=args.remaining_tasks)
    save_initial_background(background, args.output_dir)
    print(json.dumps(background.summary(), ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
