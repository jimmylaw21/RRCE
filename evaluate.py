import json

from rrce.config import config_from_args, config_parser


def main():
    parser = config_parser("Evaluate the RRCE main experiment protocol")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("erased", "retain", "cce", "ud", "coco"),
        default=["erased", "retain", "cce", "ud", "coco"],
    )
    parser.add_argument(
        "--checkpoint", help="U-Net state dictionary; defaults to training.output/unet.pt"
    )
    parser.add_argument(
        "--metrics-only", action="store_true", help="Score completed image generation"
    )
    args = parser.parse_args()
    config = config_from_args(args)
    from rrce.evaluation.runner import run_evaluation

    result = run_evaluation(config, args.splits, args.checkpoint, args.metrics_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
