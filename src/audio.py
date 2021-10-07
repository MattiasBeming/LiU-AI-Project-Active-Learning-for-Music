from pydub import AudioSegment
import pyaudio

from threading import Thread, Lock

CHUNK = 1024
player = None

running = False
paused = False
mutex = Lock()
worker = None


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
    global running
    resume()
    running = False


def deinit():
    player.terminate()


def _stream_clip(stream, clip):
    global running, worker

    # Data offset
    offset = 0

    # Stream audio
    running = True
    while running and offset < len(clip.raw_data):
        mutex.acquire()
        stream.write(clip.raw_data[offset:(offset+CHUNK-1)])
        offset += CHUNK
        mutex.release()
    running = False

    # Stop stream
    stream.stop_stream()
    stream.close()

    # Remove worker
    worker = None
