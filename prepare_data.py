from rrce.config import config_from_args, config_parser


def main():
    parser = config_parser("Prepare RRCE experiment inputs")
    parser.add_argument("--stage", choices=("manifests", "retain", "gallery"), required=True)
    config = config_from_args(args := parser.parse_args())
    from rrce.data.assets import prepare_cce_gallery, prepare_retain_latents
    from rrce.data.manifests import prepare_manifests

    stages = {
        "manifests": prepare_manifests,
        "retain": prepare_retain_latents,
        "gallery": prepare_cce_gallery,
    }
    stages[args.stage](config)


if __name__ == "__main__":
    main()
