import PySimpleGUI as sg
import numpy as np
from pathlib import Path

from api import audio as ap

#############
# CONSTANTS #
#############

# Query prior
INPUT_RANGE = (-1, 1)  # Slider limits
INPUT_RESOLUTION = .001  # Slider resolution
INDICATOR_SIZE = 10
PRIOR_CANVAS_WIDTH = 250
PRIOR_CANVAS_HEIGHT = 200
EMO_LABELS = [
    "Excited", "Delighted", "Happy",
    "Content", "Relaxed", "Calm",
    "Tired", "Bored", "Depressed",
    "Frustrated", "Angry", "Tense"
]

# Query dynamic
EVENT_PLAY = "PLAY"
EVENT_PAUSE = "PAUSE"

DYNAMIC_CANVAS_WIDTH = 550
DYNAMIC_CANVAS_HEIGHT = 120

LIMIT_LABELS_AROUSAL = ["Active / Aroused", "Passive / Calm"]
LIMIT_LABELS_VALENCE = ["Positive", "Negative"]

QUERY_MODE_AROUSAL = "arousal"
QUERY_MODE_VALENCE = "valence"

#####################
# LAYOUT GENERATORS #
#####################


def _layout_query_prior(): return [
    [
        sg.Column([
            [sg.Text("Please enter your current emotional state:",
                     key="-TITLE-")],
            [
                sg.Text("Arousal:"),
                sg.Slider(key="-AROUSAL-",
                          default_value=0,
                          range=INPUT_RANGE,
                          resolution=INPUT_RESOLUTION,
                          orientation="horizontal",
                          enable_events=True
                          )
            ],
            [
                sg.Text("Valence:"),
                sg.Slider(key="-VALENCE-",
                          default_value=0,
                          range=INPUT_RANGE,
                          resolution=INPUT_RESOLUTION,
                          orientation="horizontal",
                          enable_events=True
                          )
            ],
            [
                sg.Button("OK", key="-OK-"),
                sg.Button("Reset", key="-RESET-")
            ]
        ]),
        sg.Canvas(
            key="-CANVAS-",
            size=(PRIOR_CANVAS_WIDTH, PRIOR_CANVAS_HEIGHT),
            background_color="white"
        )
    ]
]


def _layout_query_dynamic(): return [
    [sg.Frame("Playback Controller", layout=[
        [sg.Column(layout=[[
            sg.Frame("Mode", layout=[[
                sg.Text("", key="-QUERY-MODE-", justification="center")
            ]], element_justification="center"),
            sg.Frame("File", layout=[[
                sg.Text("", key="-SONG-TITLE-", justification="center"),
            ]], element_justification="center"),
            sg.Frame("Time", layout=[[
                sg.Text("", key="-TIMER-", justification="center")
            ]], element_justification="center")
        ]], justification="center")],
        [
            sg.Slider(key="-RECORDER-",
                      default_value=0,
                      range=INPUT_RANGE,
                      resolution=INPUT_RESOLUTION,
                      orientation="vertical",
                      enable_events=True
                      ),
            sg.Canvas(
                key="-CANVAS-",
                size=(DYNAMIC_CANVAS_WIDTH, DYNAMIC_CANVAS_HEIGHT),
                background_color="white"
            )
        ],
        [
            sg.ProgressBar(
                key="-PROGRESS-",
                max_value=1.0,
                size=(50, 10),
                pad=(75, 0)
            )
        ],
        [sg.Column(layout=[[
            sg.Button("Submit", key="-OK-", disabled=True),
            sg.Button("Restart", key="-RESET-")
        ]], justification="center")]
    ])]
]

#####################
# RESULT PROCESSORS #
#####################


def process_result_default(result):
    return result


def process_result_dynamic(result):
    # Extract parameters
    sample_rate = result[0]
    points = np.array(result[1])

    # Calculate period
    p = 1.0 / sample_rate

    # Calculate end point
    end_point = points[-1, 0]
    end = end_point - (end_point % p)

    # Generate sample space from period
    x = np.linspace(0, end, int(end / p))

    # Produce sampled output
    return np.hstack((
        np.reshape(x, (-1, 1)),
        np.reshape(np.interp(x, points[:, 0], points[:, 1]), (-1, 1))
    ))

####################
# EVENT PROCESSORS #
####################


def handle_query_prior(window, event, values, emotion_indicator):

    # Skip if timeout
    if event == sg.TIMEOUT_EVENT:
        return False, None

    # Closure
    close = False

    # Check for OK event
    if event == "-OK-":
        close = True

    # Check for reset event
    elif event == "-RESET-":
        window.Element("-AROUSAL-").Update(0)
        window.Element("-VALENCE-").Update(0)
        values["-AROUSAL-"] = 0
        values["-VALENCE-"] = 0

    # Calculate position of emotion indicator
    ei_pos = (
        (values["-VALENCE-"] + 1) * PRIOR_CANVAS_WIDTH / 2,
        (-values["-AROUSAL-"] + 1) * PRIOR_CANVAS_HEIGHT / 2
    )

    window["-CANVAS-"].TKCanvas.coords(
        emotion_indicator,
        ei_pos[0] - INDICATOR_SIZE / 2,
        ei_pos[1] - INDICATOR_SIZE / 2,
        ei_pos[0] + INDICATOR_SIZE / 2,
        ei_pos[1] + INDICATOR_SIZE / 2
    )

    return close, (values["-AROUSAL-"], values["-VALENCE-"])


def handle_query_dynamic(window, event, values, file_path: Path,
                         sample_rate, points: list):

    # Update timer
    total, elapsed = ap.time_info()

    window["-TIMER-"].update(
        f"{elapsed:.2f}s / {total:.2f}s"
        if total > 0 else "0.00s / 0.00s"
    )

    window["-PROGRESS-"].update(
        elapsed / total if total > 0 else total
    )

    # Add recorder point
    if values is not None and "-RECORDER-" in values:
        points.append((elapsed, values["-RECORDER-"]))
    elif len(points) > 0:
        points.append((elapsed, points[-1][1]))

    # Plot new line
    if len(points) > 1:
        window["-CANVAS-"].TKCanvas.create_line(
            DYNAMIC_CANVAS_WIDTH * points[-2][0] / total if total > 0 else 0,
            DYNAMIC_CANVAS_HEIGHT * (1 - points[-2][1]) / 2,
            DYNAMIC_CANVAS_WIDTH * points[-1][0] / total if total > 0 else 0,
            DYNAMIC_CANVAS_HEIGHT * (1 - points[-1][1]) / 2,
            width=3, fill="red", tags="line"
        )

    # Check for completion
    window["-OK-"].update(disabled=not (total >
                          0 and np.abs(total - elapsed) < .1))

    # Break on timeout event
    if event == sg.TIMEOUT_EVENT:
        return False, None

    # Check for play event
    if event == f"-RECORDER-{EVENT_PLAY}":
        if ap.is_paused():
            ap.resume()
        else:
            points.clear()
            window["-CANVAS-"].TKCanvas.delete("line")
            ap.play(file_path)

    # Check for pause event
    elif event == f"-RECORDER-{EVENT_PAUSE}":
        if ap.is_paused():
            ap.resume()
        else:
            ap.pause()

    # Check for reset event
    elif event == "-RESET-":
        points.clear()
        window["-CANVAS-"].TKCanvas.delete("line")
        window["-PROGRESS-"].update(0)
        ap.stop()

    # Check for OK event
    elif event == "-OK-":
        return True, (sample_rate, points)

    return False, (sample_rate, points)


#######################
# WINDOW INITIALIZERS #
#######################


def init_query_prior():

    # Create window
    window = sg.Window('Emotional Prior', _layout_query_prior(),
                       return_keyboard_events=True, finalize=True)

    # Create emotional labels
    for i, lab in enumerate(EMO_LABELS):

        # Calculate geometries
        c = (PRIOR_CANVAS_WIDTH / 2, PRIOR_CANVAS_HEIGHT / 2)  # Center
        r = PRIOR_CANVAS_HEIGHT * .45  # Radius
        a = np.pi / 12 * (2 * i + 1)  # Angle

        # Create emotional labels
        window["-CANVAS-"].TKCanvas.create_text(
            c[0] + r * np.sin(a),
            c[1] - r * np.cos(a),
            justify=sg.tk.CENTER,
            text=lab
        )

    # Create emotional indicator
    emotion_indicator = window["-CANVAS-"].TKCanvas.create_oval(
        PRIOR_CANVAS_WIDTH / 2 - INDICATOR_SIZE / 2,
        PRIOR_CANVAS_HEIGHT / 2 - INDICATOR_SIZE / 2,
        PRIOR_CANVAS_WIDTH / 2 + INDICATOR_SIZE / 2,
        PRIOR_CANVAS_HEIGHT / 2 + INDICATOR_SIZE / 2
    )

    return window, emotion_indicator


def init_query_dynamic(file_path: Path, mode: str, sample_rate):

    # Create window
    window = sg.Window('Dynamic Emotion Capture', _layout_query_dynamic(),
                       return_keyboard_events=True, finalize=True)

    # Set song title
    window["-SONG-TITLE-"].update(f"{file_path.name}")

    # Set query mode
    window["-QUERY-MODE-"].update("Arousal" if mode.lower()
                                  == QUERY_MODE_AROUSAL else "Valence")

    # Add upper limit label
    window["-CANVAS-"].TKCanvas.create_text(
        0, 0,
        anchor=sg.tk.NW,
        text=(LIMIT_LABELS_AROUSAL if mode ==
              QUERY_MODE_AROUSAL else LIMIT_LABELS_VALENCE)[0],
        tags="label"
    )

    # Add lower limit label
    window["-CANVAS-"].TKCanvas.create_text(
        0, DYNAMIC_CANVAS_HEIGHT,
        anchor=sg.tk.SW,
        text=(LIMIT_LABELS_AROUSAL if mode ==
              QUERY_MODE_AROUSAL else LIMIT_LABELS_VALENCE)[1],
        tags="label"
    )

    # Bind playback events
    window["-RECORDER-"].bind("<Button-1>", EVENT_PLAY)
    window["-RECORDER-"].bind("<ButtonRelease-1>", EVENT_PAUSE)

    return window, file_path, sample_rate, []
