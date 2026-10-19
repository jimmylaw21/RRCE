from rrce.config import config_from_args, config_parser


def main():
    parser = config_parser("Train standard RRCE")
    parser.add_argument("--resume", help="Resume from a completed defense-block checkpoint")
    args = parser.parse_args()
    config = config_from_args(args)
    from rrce.training.trainer import RRCETrainer

    RRCETrainer(config).run(resume=args.resume)


if __name__ == "__main__":
    main()
