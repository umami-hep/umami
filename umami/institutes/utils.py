from distutils.spawn import find_executable
import os
import subprocess
import shlex
from umami.configuration import logger


def is_tool_available(tool):
    """Check whether `tool` is on PATH."""
    return find_executable(tool) is not None


def is_qsub_available():
    """Check whether `qsub` is available on Zeuthen."""
    return is_tool_available("/usr/gridengine/scripts/qsub")


def submit_zeuthen(args):
    """Submit job using `qsub` to Zeuthen batch system."""
    logger.info("Submitting job to Zeuthen batch system...")
    train_job_path = f"{os.path.dirname(__file__)}/zeuthen/train_job.sh"
    processed_args = build_args(args)
    qsub = "/usr/gridengine/scripts/qsub"
    try:
        subprocess.Popen([qsub, train_job_path, *processed_args]).communicate()
    except OSError as e:
        raise logger.error(str(e))


def build_args(args):
    """Build arguments as an array."""
    expanded_args = (
        f'{f" -c {args.config_file}" if args.config_file is not None else ""}'
        f'{f" -m {args.model_name}" if args.model_name is not None else ""}'
        f'{f" -e {args.epochs}" if args.epochs is not None else ""}'
        f'{" -t dl1" if hasattr(args, "dl1") else ""}'
        f'{" -t dips" if hasattr(args, "dips") else ""}'
        f'{" -t umami" if hasattr(args, "umami") else ""}'
    )
    return shlex.split(expanded_args)
