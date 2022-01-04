# Project Setup

The Python versions used and tested in the project are `3.8.1` and `3.9.5`, but other version might work as well.
**Note**: To use `Tensorflow` (used for feature extraction) a Python version of `3.7-3.9` is required as of Jan 2022.

## Linux & Mac OS

Before proceding it is recommended to install the `libasound-dev` system package (or `portaudio` for Mac OS), since `PyAudio` cannot compile without it.

Once this is done, the following steps should be performed:

```bash
# Clone the project
git clone git@github.com:MattiasBeming/LiU-AI-Project-Active-Learning-for-Music.git

# Change to project directory
cd LiU-AI-Project-Active-Learning-for-Music

# Create a virutal environment
python -m venv venv

# Activate the environment
source venv/bin/activate # may be 'venv/Scripts/activate'

# Install required packages
pip3 install -r requirements.txt
```

## Windows

Windows users face the same problem that Linux users had with `PyAudio`. Instead of installing the library directly, it is therefore recommended to install the `pipwin` package. This package may then be used to install pre-compiled Windows binaries for `PyAudio`.

Using `PowerShell`:

```bash
# Clone the project
git clone git@github.com:MattiasBeming/LiU-AI-Project-Active-Learning-for-Music.git

# Change to project directory
cd LiU-AI-Project-Active-Learning-for-Music

# Create a virutal environment
python -m venv venv

# Activate the environment
./venv/Scripts/Activate.ps1

# Install pipwin and pyaudio (skip if you want pip to compile the binaries on its own)
pip3 install pipwin==0.5.1
pipwin install pyaudio==0.2.11

# Install remaining packages
pip3 install -r requirements.txt
```

# FFMPEG Setup

To allow different audio file formats the python package `pydub` is used. For playing mp3-files however, it is important that a valid mpeg-decoder is installed on the system. The `pydub` package defaults to using `libav` or `ffmpeg` if their binaries can be found on the system.

To set this up properly for different platforms, follow the instructions on [pydub's GitHub page](https://github.com/jiaaro/pydub#getting-ffmpeg-set-up).
