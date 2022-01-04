from collections import abc
from pathlib import Path
import time
import re

from threading import Thread
from api import audio
from api import gui

from init_phase import init_phase
from viability_phase import viability_phase
from evaluation_phase import create_datetime_subdir, evaluation_phase
from evaluation_phase import load_all_learning_profiles
from phase_utils import retrieve_best_learning_profiles
from user_phase import user_phase


def _phase_header(header):
    n = len(header)
    print()
    print("#" * (n + 8))
    print(f"# {'=' * (n + 4)} #")
    print(f"# = {header} = #")
    print(f"# {'=' * (n + 4)} #")
    print("#" * (n + 8))
    print()


def _pick_learning_profiles(learning_profiles: list):

    # Show info
    n = len(learning_profiles)
    print(f"There are {n} Learning Profiles.")

    # Prompt for starting index
    start_index = -1
    print(f"Pick a starting index between 0 and {n-1}:")
    while True:
        try:
            start_index = int(input("> "))
            if start_index >= 0 or start_index < n:
                break
        except ValueError:
            continue

    # Prompt for stopping index
    stop_index = -1
    print(f"Pick a stopping index between {start_index} and {n-1}:")
    while True:
        try:
            stop_index = int(input("> "))
            if stop_index >= start_index or stop_index < n:
                break
        except ValueError:
            continue

    return learning_profiles[start_index:(stop_index+1)]


def _pick_multiple_learning_profiles(learning_profiles: list):

    # Initial prompt
    print("Pick what Learning Profiles to evaluate.")

    indexed_lps = {i: lp for i, lp in enumerate(learning_profiles)}

    picked_inds = []
    while True:
        # Print unpicked LPs
        print("Learning Profiles to pick from:")
        if len(picked_inds) == len(indexed_lps):
            print("\t-")
        else:
            for i, lp in indexed_lps.items():
                if i not in picked_inds:
                    print(f"\t{i}: {lp.get_id()}")

        # Print picked LPs
        print("Picked Learning Profiles:")
        if not picked_inds:
            print("\t-")
        else:
            for i in sorted(picked_inds):
                print(f"\t{i}: {indexed_lps[i].get_id()}")

        # Input prompt
        print("Enter indices on format 'i' or 'i-j'.")
        print("Drop staged Learning Profiles with 'drop i'.")
        print("Write 'done' when you are done.")

        # Handle input
        try:
            idx = input("> ")
            if idx == "done":  # Check if done
                break
            elif bool(re.match("^[0-9]+-[0-9]+$", idx)):  # Check if range
                span_str = idx.split("-")
                picked_inds += [i for i in range(
                    int(span_str[0]), int(span_str[1]) + 1)
                    if i not in picked_inds]
            elif bool(re.match("^drop [0-9]+$", idx)):
                picked_inds.remove(int(idx.split()[1]))
            elif int(idx) in indexed_lps.keys() \
                    and int(idx) not in picked_inds:  # Check if singular
                picked_inds.append(int(idx))
        except ValueError:
            continue

    return [indexed_lps[i] for i in picked_inds]


def _nested_dict_ids(nested):
    for _, value in nested.items():
        if isinstance(value, abc.Mapping):
            yield from _nested_dict_ids(value)
        elif isinstance(value, abc.Iterable):
            for lp in value:
                yield lp.get_id()
        else:
            raise ValueError(f"Invalid structure (value was '{value}')")


def _best_learning_profiles(input_dir: Path, learning_profiles: list,
                            n_models_per_cat: int):

    # Load learning profile descriptions and choose best
    lp_descs = load_all_learning_profiles(input_dir)
    lp_descs_best = retrieve_best_learning_profiles(lp_descs, n_models_per_cat)

    # Use descriptions to retrieve actual learning profiles
    return [lp for lp in learning_profiles
            if lp.get_id() in _nested_dict_ids(lp_descs_best)]


def model_selection_process(data_dir: Path, output_dir: Path,
                            sliding_window_length: int,
                            batch_size: int, num_iterations: int,
                            seed_percent: float, n_threads: int):
    """
    Runs the model selection process.

    Args:
        data_dir (Path): The directory where all `.csv` and
            `.npy` files are located.
        output_dir (Path): A directory where all Learning
            Profile results will be stored.
        sliding_window_length (int): The sliding window size to use.
        batch_size (int): The batch size to use.
        num_iterations (int): Number of batches to process.
        seed_percent (float): Percent of initial seed data
            to use before applying Active Learning.
        n_threads (int): The number of threads to use.
    """
    ########################
    # Initialization Phase #
    ########################
    _phase_header("INIT PHASE")
    learning_profiles = init_phase(
        data_dir,
        sliding_window_length=sliding_window_length,
        batch_size=batch_size,
        model_eval=False
    )

    ##########
    # Filter #
    ##########
    _phase_header("LEARNING PROFILE FILTER")
    filtered_learning_profiles = _pick_learning_profiles(learning_profiles)

    ###################
    # Viability Phase #
    ###################
    _phase_header("VIABILITY PHASE")
    viability_phase(filtered_learning_profiles, num_iterations,
                    seed_percent, n_threads)

    ####################
    # Evaluation Phase #
    ####################
    _phase_header("EVALUATION PHASE")
    stamped_output_dir = create_datetime_subdir(output_dir)
    evaluation_phase(stamped_output_dir, filtered_learning_profiles)

    print("Evaluated successfully!")

    # Done
    _phase_header("DONE")


def model_evaluation_process(data_dir: Path, input_dir: Path, output_dir: Path,
                             audio_dir: Path, sliding_window_length: int,
                             batch_size: int, num_iterations: int,
                             seed_percent: float, audio_file_ext: str,
                             n_models_per_cat: int):
    """
    Runs the model evaluation process.

    Args:
        data_dir (Path): The directory where all `.csv` and
            `.npy` files are located.
        input_dir (Path): A directory with Learning Profile results from
            the model_selection process.
        output_dir (Path): A directory where all Learning
            Profile results will be stored.
        audio_dir (Path): A directory where all audio files are located.
        sliding_window_length (int): The sliding window size to use.
        batch_size (int): Number of batches to process.
        num_iterations (int): Number of batches to process.
        seed_percent (float): Percent of initial seed data
            to use before applying Active Learning.
        audio_file_ext (str): File extension of the audio files
            in `data_dir`.
        n_models_per_cat (int): Number of models per category item
            (AL/ML method etc.).

    Raises:
        FileNotFoundError: If `input_dir` is not a valid directory.
    """

    ########################
    # Initialization Phase #
    ########################
    _phase_header("INIT PHASE")
    learning_profiles = init_phase(
        data_dir,
        sliding_window_length=sliding_window_length,
        batch_size=batch_size,
        model_eval=True
    )

    ##########
    # Filter #
    ##########
    _phase_header("LEARNING PROFILE FILTER")

    # Validity check
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: '{input_dir}'")

    # Get best learning profiles
    filtered_learning_profiles = _best_learning_profiles(
        input_dir, learning_profiles, n_models_per_cat)

    # Pick what learning profiles to evaluate
    picked_learning_profiles = _pick_multiple_learning_profiles(
        filtered_learning_profiles)

    ##############
    # User Phase #
    ##############
    _phase_header("USER PHASE")

    # Initialize audio
    audio.init()

    # User phase wrapper
    def _user_phase_thread_func():
        for _ in user_phase(
                picked_learning_profiles, audio_dir,
                num_iterations, seed_percent,
                audio_file_ext):
            pass

    # Start application
    _app = Thread(target=_user_phase_thread_func)

    print("Starting User Phase thread...")
    _app.start()

    # Drive GUI
    while _app.is_alive():
        time.sleep(.01)  # Allow other threads to breathe
        gui.update_windows()
    print("The GUI loop on main thread was exited " +
          "since the User Phase thread was stopped!")

    # Exit GUI
    print("Destroying GUI...")
    gui.destroy()
    print("The GUI was successfully destroyed!")

    # Deinitialize audio
    audio.deinit()

    ####################
    # Evaluation Phase #
    ####################
    _phase_header("EVALUATION PHASE")

    stamped_output_dir = create_datetime_subdir(output_dir)
    evaluation_phase(stamped_output_dir, picked_learning_profiles)

    print("Evaluated successfully!")

    # Done
    _phase_header("DONE")


def presentation_process():
    pass
