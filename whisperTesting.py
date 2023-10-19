import whisper
import pprint
import subprocess as sp
import os


def slide_window(model, path):
    windows = whisper.load_audio(path)
    while len(windows) > 0:
        audio = whisper.pad_or_trim(windows)
        result = whisper.transcribe(model, audio, no_speech_threshold=0.5, word_timestamps=True)
        print(pprint.pformat(result))
        #Write the text to the results directory
        with open("results/" + path + ".txt", "a") as f:
            f.write(result["text"] + "\n")
        windows = windows[len(audio):] # evil sliding window???


model = whisper.load_model("small")
files = ["frank_guglielmo_edgetest_session.mp4"]
# files = ["englishAndItalianAudio.m4a"]
for file in files:
    slide_window(model, file)

#Delete the files in the results directory
# for file in os.listdir("results/"):
#     os.remove("results/" + file)
