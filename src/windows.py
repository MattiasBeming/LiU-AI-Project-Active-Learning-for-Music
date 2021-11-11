import PySimpleGUI as sg
import math

#############
# CONSTANTS #
#############

# Query prior
INPUT_RANGE = (-1, 1)  # Slider limits
INPUT_RESOLUTION = .001  # Slider resolution
INDICATOR_SIZE = 10
CANVAS_WIDTH = 250
CANVAS_HEIGHT = 200
EMO_LABELS = [
    "Excited", "Delighted", "Happy",
    "Content", "Relaxed", "Calm",
    "Tired", "Bored", "Depressed",
    "Frustrated", "Angry", "Tense"
]

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
        sg.Canvas(key="-CANVAS-", size=(CANVAS_WIDTH, CANVAS_HEIGHT),
                  background_color="white")
    ]
]

####################
# EVENT PROCESSORS #
####################


def handle_query_prior(window, event, values, emotion_indicator):

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
        (values["-VALENCE-"] + 1) * CANVAS_WIDTH / 2,
        (-values["-AROUSAL-"] + 1) * CANVAS_HEIGHT / 2
    )

    window["-CANVAS-"].TKCanvas.coords(
        emotion_indicator,
        ei_pos[0] - INDICATOR_SIZE / 2,
        ei_pos[1] - INDICATOR_SIZE / 2,
        ei_pos[0] + INDICATOR_SIZE / 2,
        ei_pos[1] + INDICATOR_SIZE / 2
    )

    return close, (values["-AROUSAL-"], values["-VALENCE-"])


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
        c = (CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)  # Center
        r = CANVAS_HEIGHT * .45  # Radius
        a = math.pi / 12 * (2 * i + 1)  # Angle

        # Create emotional labels
        window["-CANVAS-"].TKCanvas.create_text(
            c[0] + r * math.sin(a),
            c[1] - r * math.cos(a),
            justify=sg.tk.CENTER,
            text=lab
        )

    # Create emotional indicator
    emotion_indicator = window["-CANVAS-"].TKCanvas.create_oval(
        CANVAS_WIDTH / 2 - INDICATOR_SIZE / 2,
        CANVAS_HEIGHT / 2 - INDICATOR_SIZE / 2,
        CANVAS_WIDTH / 2 + INDICATOR_SIZE / 2,
        CANVAS_HEIGHT / 2 + INDICATOR_SIZE / 2
    )

    return window, emotion_indicator
