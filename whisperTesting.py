import whisper
import pprint
import subprocess as sp
import os


def splitAudio(audioFile):
    # split audio into 30 second chunks
    sp.run(f"ffmpeg -i {audioFile} -f segment -segment_time 30 -c copy ./audioFiles/{audioFile}%03d.m4a".split())
    # sp.run("ffmpeg -i englishAndItalianAudio.m4a -f segment -segment_time 30 -c copy ./audioFiles/audio%09d.m4a".split())

def transcribeAll():
    audio_directory = "./audioFiles"

    for filename in os.listdir(audio_directory):
        if filename.endswith(".m4a"):
            transcribe(f"{audio_directory}/{filename}")
            continue
        else:
            continue

    #Delete all files in audioFiles directory
    for filename in os.listdir(audio_directory):
        if filename.endswith(".m4a"):
            os.remove(f"{audio_directory}/{filename}")
            continue
        else:
            continue





def transcribe(audioFile):
    model = whisper.load_model("base")

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audioFile)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)

    audioFileRelativePath = audioFile.split("/")[-1]

    #Write the result to a text file named afeter the audio file
    with open(f"./results/{audioFileRelativePath}.txt", "w") as text_file:
        text_file.write(pprint.pformat(result))

    # print the recognized text
    print(result.text)

# splitAudio()

# transcribe("audioFiles/audio000000000.m4a")
# transcribe("audioFiles/audio000000001.m4a")

splitAudio("englishAndItalianAudio.m4a")
transcribeAll()

# remove all the files in results directory
# for filename in os.listdir("./results"):
#     if filename.endswith(".txt"):
#         os.remove(f"./results/{filename}")
#         continue
#     else:
#         continue


