import whisper
import pprint
import subprocess as sp


def slide_window(model, path):
    windows = whisper.load_audio(path)
    while len(windows) > 0:
        audio = whisper.pad_or_trim(windows)
        result = whisper.transcribe(model, audio)
        print(pprint.pformat(result))
        windows = windows[len(audio):] # evil sliding window???


model = whisper.load_model("base")
files = ["englishAndItalianAudio.m4a"]
# files = ["englishAndItalianAudio.m4a"]
for file in files:
    slide_window(model, file)