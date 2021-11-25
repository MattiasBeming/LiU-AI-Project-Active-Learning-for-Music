from threading import Thread
from api import audio
from api import gui

#################
# MAIN FUNCTION #
#################


def main():
    pass


#########################################
# START APPLICATION & RUN MAIN FUNCTION #
#########################################

# Initialize audio
audio.init()

# Start application
_app = Thread(target=main)
print("Starting application thread...")
_app.start()
print("Application thread started!")

# Drive GUI
print("Entering GUI loop on main thread.")
while _app.is_alive():
    gui.update_windows()
print("The GUI loop on main thread was exited " +
      "since the application thread was stopped!")

# Exit GUI
print("Destroying GUI...")
gui.destroy()
print("The GUI was successfully destroyed!")

# Deinitialize audio
audio.deinit()
