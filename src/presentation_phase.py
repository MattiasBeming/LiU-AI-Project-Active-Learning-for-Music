from matplotlib import pyplot as plt
from phase_utils import Eval, Pres


def presentation_phase(learning_profiles=[], eval=Eval.AROUSAL, nr_models=-1):
    """
    Plot the best 'nr_models' from the learning profiles
    for the given evaluation mode.

    Args:
        learning_profiles (list): List of learning profiles (using lpParser).
        eval (Enum): method of evaluation. Defaults to Eval.AROUSAL.
        nr_models (int): number of models to include in plot.
            Defaults to -1 (All models included).
    """
    if not learning_profiles:
        raise ValueError("No learning profiles given")

    plt.style.use('ggplot')

    # Set eval mode for all learning profiles
    [lp.set_eval_mode(eval) for lp in learning_profiles]

    # Sort according to best score (acending)
    sorted_LPs = sorted(
        learning_profiles, key=lambda lp: lp.get_score(), reverse=True)

    # Select 'nr_models' models with the highest score
    min_ = min(len(sorted_LPs), nr_models)
    if nr_models > -1:
        sorted_LPs = sorted_LPs[-min_:]

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


def get_specific_learning_profiles(learning_profiles=[], pres=Pres.AL):
    """
    Yield a list of learning profiles for every attribute for
    Presentation mode.

    Args:
        learning_profiles (list): List of learning profiles (using lpParser).
        pres (Enum): presentation mode. Defaults to Pres.AL.
    """
    # Set pres mode for all learning profiles
    [lp.set_pres_mode(pres) for lp in learning_profiles]

    # Loop through all attributes and yield the models with the same attribute
    for attr in set([lp.get_attr() for lp in learning_profiles]):
        # Retrieve all learning profiles with the same attribute
        yield [lp for lp in learning_profiles if lp.get_attr() == attr]
