from pathlib import Path

from rrce.config import config_from_args, config_parser


def main():
    parser = config_parser("Generate images from a prompt manifest")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sampling", choices=("erased", "retain", "coco"), default="erased")
    args = parser.parse_args()
    config = config_from_args(args)
    from rrce.evaluation.sampling import generate_images
    from rrce.io import read_manifest, write_json
    from rrce.models.diffusion import load_inference_pipeline

    rows = read_manifest(args.manifest)
    sampling = config.evaluation.sampling[args.sampling]
    pipeline = load_inference_pipeline(config, sampling, args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    generate_images(pipeline, rows, args.output, config, sampling)
    write_json(args.output / "config.json", config.to_dict())


if __name__ == "__main__":
    main()
