from matplotlib import pyplot as plt
from phase_utils import Eval, LearningProfileDescription, sort_by_score


def presentation_phase(learning_profiles=[], eval=Eval.AROUSAL, nr_models=-1):
    """
    Plot the best 'nr_models' from the learning profiles
    for the given evaluation mode.

    Args:
        learning_profiles (list): List of learning profiles
            (using LearningProfileDescription).
        eval (Enum): method of evaluation. Defaults to Eval.AROUSAL.
        nr_models (int): number of models to include in plot.
            Defaults to -1 (All models included).
    """
    if not learning_profiles:
        raise ValueError("No learning profiles given")

    sorted_LPs = sort_by_score(learning_profiles, eval, nr_models)

    plt.style.use('ggplot')
    # Plot all learning profiles
    # Max comparable models before colors run out = 40
    COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    NUM_COLORS = len(COLORS)
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)

    # Get length of MSE
    x = range(1, len(sorted_LPs[0].get_MSE_arousal()) + 1)
    for i, lp in enumerate(sorted_LPs):
        plt.plot(x, lp.get_MSE(), label=f"{lp.get_name(True, True)}",
                 color=COLORS[i % NUM_COLORS],
                 linestyle=LINE_STYLES[i % NUM_STYLES],
                 linewidth=2.0)

    plt.legend()
    plt.title(f"MSE {eval.name} over each AL iteration")
    plt.ylabel("MSE")
    plt.xlabel(f"Batch")
    plt.show()
