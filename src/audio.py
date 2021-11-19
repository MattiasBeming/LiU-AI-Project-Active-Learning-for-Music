from pydub import AudioSegment
import pyaudio

from threading import Thread, Lock

CHUNK = 1024
player = None

running = False
paused = False
mutex = Lock()
worker = None

_clip_time_total = 0
_clip_time_elapsed = 0


def init():
    global player
    player = pyaudio.PyAudio()


def play(file_path):
    global worker

    # Stop current song
    if (running and worker is not None):
        stop()
        worker.join()

    # Load audio clip
    clip = AudioSegment.from_file(file_path)

    # Start stream
    format = player.get_format_from_width(clip.sample_width)
    stream = player.open(format=format,
                         channels=clip.channels,
                         rate=clip.frame_rate,
                         output=True)

    # Create worker
    worker = Thread(target=_stream_clip, args=(stream, clip))

    # Start worker
    worker.start()


def pause():
    global paused
    if paused or not running:
        return
    paused = True
    mutex.acquire()


def resume():
    global paused
    if not paused:
        return
    mutex.release()
    paused = False


def is_paused():
    return paused


def stop():
    global running, _clip_time_total, _clip_time_elapsed
    resume()
    _clip_time_total = 0
    _clip_time_elapsed = 0
    running = False


def deinit():
    stop()
    player.terminate()
    if worker is not None:
        worker.join()


def time_info():
    """
    Returns the full clip length in seconds, and the current elapsed time.

    Returns:
        (float, float): The full clip length in seconds, and the elapsed time.
    """
    return _clip_time_total, _clip_time_elapsed


def _stream_clip(stream, clip):
    global running, worker, _clip_time_total, _clip_time_elapsed

    # Data offset
    offset = 0

    # Time handling
    _clip_time_total = clip.duration_seconds
    _clip_time_elapsed = 0

    # Current chunk
    current_chunk = b""

    # Stream audio
    running = True
    while running and offset < len(clip.raw_data):

        # Stream current chunk
        stream.write(current_chunk)

        # Read next chunk
        mutex.acquire()
        current_chunk = clip.raw_data[offset:(offset+CHUNK-1)]
        offset += CHUNK
        _clip_time_elapsed = offset / (clip.sample_width * clip.frame_rate)
        mutex.release()
    running = False

    # Stop stream
    stream.stop_stream()
    stream.close()

    # Remove worker
    worker = None
