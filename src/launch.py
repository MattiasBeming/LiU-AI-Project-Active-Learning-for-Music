from argparse import ArgumentParser
from pathlib import Path
import time
import sys

from processes import model_selection_process
from processes import model_evaluation_process
from processes import presentation_process

# Suppress sklearn warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

###################################################
# Process wrappers for specific argument handling #
###################################################


def _ms_wrapper(name, argv):
    # Construct model selection parser
    parser = ArgumentParser(name)
    parser.add_argument("data_dir", type=str,
                        help="a directory where all .csv and "
                        ".npy files are located")
    parser.add_argument("-o", "--output-dir", metavar="path",
                        type=str, default="out/model_selection",
                        help="a directory where all Learning "
                        "Profile results will be stored (default: '"
                        "out/model_selection')")
    parser.add_argument("-sw", "--sliding-window", dest='sw',
                        metavar="n",
                        type=int, default=5,
                        help="the sliding window size to use (default: 5)")
    parser.add_argument("-bs", "--batch-size", dest='bs',
                        metavar="n",
                        type=int, default=100,
                        help="the batch size to use (default: 100)")
    parser.add_argument("-ni", "--num-iterations", dest='ni',
                        metavar="n",
                        type=int, default=-1,
                        help="number of batches to process. If -1, all "
                        "Learning Profiles will be depleted (default: -1)")
    parser.add_argument("-sp", "--seed-percent", dest='sp',
                        metavar="f",
                        type=float, default=0.1,
                        help="percent of initial seed data to use before "
                        "applying Active Learning (default: 0.1)")
    parser.add_argument("-nt", "--n-threads", dest='nt',
                        metavar="n",
                        type=int, default=1,
                        help="number of threads to use (default: 1)")

    # Parse arguments
    args = parser.parse_args(argv)

    # Execute model selection process
    model_selection_process(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        sliding_window_length=args.sw,
        batch_size=args.bs,
        num_iterations=args.ni,
        seed_percent=args.sp,
        n_threads=args.nt
    )


def _me_wrapper(name, argv):
    parser = ArgumentParser(name)


def _pp_wrapper(name, argv):
    parser = ArgumentParser(name)


##########################
# Core argument handling #
##########################

# Process names
processes = {
    "model_selection": _ms_wrapper,
    "model_evaluation": _me_wrapper,
    "presentation": _pp_wrapper
}

# Parser core arguments
parser = ArgumentParser()
parser.add_argument("process",
                    choices=processes.keys(),
                    help="the process to run")
core_args = parser.parse_args(sys.argv[1:2])

# Evoke appropriate wrapper
t = time.time()
processes[core_args.process](core_args.process, sys.argv[2:])
print(f"Took {time.time() - t:.1f} seconds.")
