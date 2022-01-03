from pathlib import Path
from init_phase import init_phase
from viability_phase import viability_phase
from evaluation_phase import create_datetime_subdir, evaluation_phase


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
            if start_index < 0 or start_index >= n:
                continue
        except ValueError:
            continue
        break

    # Prompt for stopping index
    stop_index = -1
    print(f"Pick a stopping index between {start_index} and {n-1}:")
    while True:
        try:
            stop_index = int(input("> "))
            if stop_index < start_index or stop_index >= n:
                continue
        except ValueError:
            continue
        break

    return learning_profiles[start_index:(stop_index+1)]


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

    # Initialization Phase
    _phase_header("INIT PHASE")
    learning_profiles = init_phase(
        data_dir,
        sliding_window_length=sliding_window_length,
        batch_size=batch_size,
        model_eval=False
    )

    # Filter
    _phase_header("LEARNING PROFILE FILTER")
    filtered_learning_profiles = _pick_learning_profiles(learning_profiles)

    # Viability Phase
    _phase_header("VIABILITY PHASE")
    viability_phase(filtered_learning_profiles, num_iterations,
                    seed_percent, n_threads)

    # Evaluation Phase
    _phase_header("EVALUATION PHASE")
    stamped_output_dir = create_datetime_subdir(output_dir)
    evaluation_phase(stamped_output_dir, filtered_learning_profiles)

    print("Evaluated successfully!")

    # Done
    _phase_header("DONE")


def model_evaluation_process():
    pass


def presentation_process():
    pass
