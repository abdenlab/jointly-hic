"""Main entrypoint for the jointly_hic package."""

import logging

from jointly_hic.parser import JointlyCommandLineInterface


def setup_logging(args):
    """Set up logging to file and console."""
    logger = logging.getLogger("joint_pca")
    logger.handlers.clear()
    logger.setLevel(args.log_level)

    # Setup logging to file
    handler = logging.FileHandler(args.log, mode="w", encoding="utf-8")
    handler.setLevel(args.log_level)
    formatter = logging.Formatter("%(asctime)s::%(levelname)s::%(funcName)s:   %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Setup logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(args.log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Disable propagation to parent loggers
    logger.propagate = False
    return logger


def main():
    """Run the jointly_hic program."""
    # Parse arguments as dispatch subcommand
    parser = JointlyCommandLineInterface().parser
    args = parser.parse_args()
    logger = setup_logging(args)
    try:
        logger.info("Starting jointly-hic")
        args.func(args)
        logger.info("Finished jointly-hic")
    except Exception as e:
        logger.exception("Error running jointly-hic")
        logger.exception(e)
        raise e
