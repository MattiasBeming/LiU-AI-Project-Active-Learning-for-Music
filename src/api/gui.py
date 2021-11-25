import PySimpleGUI as sg
from threading import Lock
import uuid

from api import windows

################
# WINDOW AGENT #
################


class WindowAgent:

    def __init__(self, window_initializer, event_processor,
                 result_processor, allow_native_close, init_args):
        """
        Creates a new window agent.

        Args:
            window_initializer (function): The function to use when
                initializing this agent's window. The function may take an
                arbritrary number of arguments, and the init_args list will
                be spread into it when the function is invoked from start().

                The function should create a new window and return it. More
                values may be returned after the window if a tuple is used -
                these values will then be stored in the agent, and passed to
                the event_processor when invoked through the update() function.
            event_processor (function): The function to use for handling
                window events. As input parameters, the function should take a
                window, an event, and a dict of values associated with the
                event. There may also be a final vararg parameter, where the
                returned arguments from window_initializer will be passed to
                if invoked through the update() function.

                The return-value of this function should be a tuple, where the
                first value should be a boolean, and the second value will
                be stored as the agent's result. The first value should be
                true if the window should be closed due to the event/values.
            result_processor (function): The function to use for processing
                the final result before returning it from wait_result(). As
                input it is expected to take a result on the same form as the
                outputted result from event_processor. The returned value
                should be a processed version of this result.
            allow_native_close (bool): If True, the user may close the window
                natively, i.e. by pressing the exit button on the window
                border.
            init_args (list): The arguments to send to the window_initializer.
        """
        self.ref = uuid.uuid4()
        self.window = None
        self.window_initializer = window_initializer
        self.event_processor = event_processor
        self.result_processor = result_processor
        self.allow_native_close = allow_native_close
        self.init_args = init_args
        self.mutex = Lock()
        self.result = None

        # Acquire mutex to block result-fetching
        self.mutex.acquire()

    def start(self):
        """
        Starts this agent by calling its window initializer. This method
        creates the agent's window, and should be called before usage.
        """
        wi_res = self.window_initializer(*self.init_args)

        if type(wi_res) in (tuple, list):
            self.window, *self.args = wi_res
        else:
            self.window = wi_res
            self.args = []

    def update(self, event, values):
        """
        Handles the incoming event and its values. If the event is either
        WINDOW_CLOSED or "Quit" the exit() function is invoked. Otherwise,
        the event_processor of the agent will be invoked with this agent's
        window, this method's two input arguments, and the arguments returned
        by the window initializer. The event_processor may indicate that the
        window should be closed due to how the event/values were processed,
        resulting once again in exit() being called.

        Args:
            event (str): An event to process, fetched from this agent's window.
            values (dict): Values to process, fetched from this agent's window.
        """

        # Check for exit
        if event == sg.WINDOW_CLOSED or event == "Quit":
            if self.allow_native_close:
                return self.exit()
            return

        # Process events
        close, result = self.event_processor(
            self.window, event, values, *self.args)

        # Update result if there was one
        if result is not None:
            self.result = result

        # Check for window closure
        if close:
            self.exit()

    def exit(self):
        """
        Exits this agent and closes its window.
        """

        if not self.is_alive():
            return

        self.window.close()
        self.mutex.release()

    def is_alive(self):
        """
        Checks if the agent is alive. The agent is alive if exit() has not
        yet been called.

        Returns:
            bool: true if the agent is alive, false otherwise.
        """
        return self.mutex.locked()

    def wait_result(self):
        """
        Waits for the agent to exit(), whereupon the latest result from the
        event_processor will be returned.

        Returns:
            Any: The latest result from the event_processor.
        """
        self.mutex.acquire()
        r = self.result
        self.mutex.release()
        return self.result_processor(r)


#################
# AGENT STORAGE #
#################

_store_mutex = Lock()
_agents = {}


def _new_window_agent(window_initializer, event_processor,
                      result_processor, allow_native_close, *init_args):

    agent = WindowAgent(window_initializer, event_processor,
                        result_processor, allow_native_close, init_args)

    _store_mutex.acquire()
    _agents[agent.ref] = agent
    _store_mutex.release()

    return agent


def _get_window_agent(ref):
    _store_mutex.acquire()

    agent = _agents.get(ref)

    _store_mutex.release()

    return agent


def _pop_window_agent(ref):
    _store_mutex.acquire()

    agent = _agents.get(ref)

    if agent is None:
        return None

    _agents.pop(ref)

    _store_mutex.release()

    return agent


####################
# AGENT MANAGEMENT #
####################


def update_windows():
    """
    Updates all windows by forwarding events to their window agents. A single
    event is processed per call.
    """
    global _agents

    # Get current agent references
    _store_mutex.acquire()
    refs = list(_agents.keys())
    _store_mutex.release()

    # Poll window events
    window, event, values = sg.read_all_windows(50)

    for ref in refs:
        agent = _get_window_agent(ref)

        # Check if agent was found
        if agent is None:
            continue

        # Start agent if window does not exists
        if agent.window is None:
            agent.start()
            continue

        # Skip update if agent is dead
        if not agent.is_alive():
            continue

        # Check if the current event belongs to the agent
        if event != sg.TIMEOUT_EVENT and agent.window != window:
            continue

        # Update agent with current event
        agent.update(event, values)


def destroy():
    """
    Deinitializes the GUI system. Must be called before the program is closed.
    """
    global _agents

    # Destroy all windows
    _store_mutex.acquire()

    for agent in _agents.values():
        agent.exit()

    _agents = {}

    _store_mutex.release()


def wait_result(ref):
    """
    Waits for the referenced agent to finish, whereupon its latest result will
    be returned.

    Args:
        ref (uuid.UUID): The UUID of the agent. Usually received when a window
            is created.

    Raises:
        KeyError: When an agent with the given UUID could not be found.

    Returns:
        Any: The latest result from the agent.
    """
    agent = _get_window_agent(ref)

    if agent is None:
        raise KeyError(f"Invalid agent reference \"{ref}\"")

    result = agent.wait_result()

    _pop_window_agent(ref)

    return result


def is_alive(ref):
    """
    Checks if the agent is alive. The agent is alive if it has not yet
    been closed or exited.

    Args:
        ref (uuid.UUID): The UUID of the agent. Usually received when a window
            is created.

    Returns:
        bool: true if the agent is alive, false otherwise.
    """
    agent = _get_window_agent(ref)

    if agent is None:
        return False

    return _get_window_agent(ref).alive


###########
# WINDOWS #
###########


def query_prior():
    """
    Prompts the user for their current emotional state in a new window. The
    resulting output can be fetched with wait_result(), and will be a tuple on
    the form: (float: arousal, float: valence).

    Returns:
        uuid.UUID: The uuid of the new window.
    """

    agent = _new_window_agent(
        windows.init_query_prior,
        windows.handle_query_prior,
        windows.process_result_default,
        True
    )

    return agent.ref


def query_dynamic(file_path, mode, sample_rate=2):
    """
    Opens a windows where the user must record either arousal or valence over
    time, while a song is playing. The resulting output can be fetched with
    wait_result(), where the result will be a 2-dimensional numpy array with
    two columns. The first column will contain evenly spaced time points for
    the entire song in seconds according to the specified sample_rate. The
    second column will contain the captured input values at those time points.

    Args:
        file_path (pathlib.Path): A path to the music file to play.
        mode (str): Either "arousal" or "valence".
        sample_rate (int, optional): The number of output samples per second.
            Defaults to 2.

    Raises:
        ValueError: If the specified mode was not valid.

    Returns:
        uuid.UUID: The uuid of the new window.
    """

    if mode not in (windows.QUERY_MODE_AROUSAL, windows.QUERY_MODE_VALENCE):
        raise ValueError(f"Invalid dynami query mode: '{mode}'")

    agent = _new_window_agent(
        windows.init_query_dynamic,
        windows.handle_query_dynamic,
        windows.process_result_dynamic,
        False,
        file_path, mode, sample_rate
    )

    return agent.ref
