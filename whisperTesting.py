import random
import re
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union, List

import numpy as np
import torch
import tqdm
import whisper

from whisper.audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from whisper.decoding import BeamSearchDecoder, DecodingOptions, DecodingResult
from whisper.timing import add_word_timestamps
from whisper.tokenizer import LANGUAGES, get_tokenizer
from whisper.utils import (
    exact_div,
    format_timestamp,
    make_safe,
    sys,
)

import whisper
import pprint
import subprocess as sp
import os
import torch
import difflib
import json
import uuid

torch.cuda.is_available()

transcription_map = {}


def transcribe(model, path, language, no_speech_treshold, logprob_treshold, segment_no_speech_prob,
               segment_avg_logprob):
    print(f"Transcribing audio...")
    windows = whisper.load_audio(path)
    window_index = 0

    # json_path = "results/json/" + path + "_" + str(no_speech_treshold) + "-" + str(logprob_treshold) + "_" + str(
    #  segment_no_speech_prob) + "-" + str(segment_avg_logprob) + language + ".json"
    # txt = "results/" + path + "_" + str(no_speech_treshold) + "-" + str(logprob_treshold) + "_" + str(
    #        segment_no_speech_prob) + "-" + str(segment_avg_logprob) + language + ".txt"
    if language is None:
        json_path = "results/json/" + path + "None" + ".json"
        txt = "results/" + path + "None" + ".txt"
    else:
        json_path = "results/json/" + path + language + ".json"
        txt = "results/" + path + language + ".txt"

    with open(json_path, "w") as f:
        f.write("[")

    with open(txt, "w") as f:
        f.write("")

    error_bucket = []
    while len(windows) > 0:
        audio = whisper.pad_or_trim(windows)
        result = whisper.transcribe(model, audio,
                                    language=language,
                                    word_timestamps=True,
                                    no_speech_threshold=no_speech_treshold,
                                    logprob_threshold=logprob_treshold)
        print(f"Remaining {len(windows) / len(audio)}")

        with open(json_path, "a") as f:
            if len(windows) == 1:
                f.write(json.dumps(result))
            else:
                f.write(json.dumps(result) + ",")
        with open(txt, "a", encoding="utf-8") as f:
            for segment in result['segments']:
                # if segment['no_speech_prob'] < segment_no_speech_prob:
                if segment['avg_logprob'] > segment_avg_logprob:
                    f.write(segment['text'])
                else:
                    f.write("ERROR\n")
                    f.write(segment['text'])
                    f.write("\nERROR\n")
                    error_bucket.append([segment['start'] + window_index, segment['end'] + window_index])
                # else:
                #     error_bucket.append([segment['start'] + window_index, segment['end'] + window_index])
                f.write("\n")
        window_index += 30
        windows = windows[len(audio):]  # evil sliding window???

    with open(json_path, "a") as f:
        f.write("]")

    return error_bucket


def cross_validation(model, path):
    minimum_diff = float("inf")
    best_model = {}
    for no_speech_treshold in [0.4, 0.5, 0.6]:
        for logprob_treshold in [-0.8, -0.9, -1.0]:
            for segment_no_speech_prob in [0.4, 0.5, 0.6]:
                for segment_avg_logprob in [-0.7, -0.8, -0.9]:
                    transcribe(model, path, "en", no_speech_treshold, logprob_treshold, segment_no_speech_prob,
                               segment_avg_logprob)
                    diff = get_diff(path, no_speech_treshold, logprob_treshold, segment_no_speech_prob,
                                    segment_avg_logprob)
                    if minimum_diff > diff:
                        minimum_diff = diff
                        best_model = {"no_speech_treshold": no_speech_treshold, "logprob_treshold": logprob_treshold,
                                      "segment_no_speech_prob": segment_no_speech_prob,
                                      "segment_avg_logprob": segment_avg_logprob}
    print(best_model)
    print(minimum_diff)
    return best_model


def get_diff(path, no_speech_treshold, logprob_treshold, segment_no_speech_prob, segment_avg_logprob):
    txt = "results/" + path + "[" + str(no_speech_treshold) + ":" + str(logprob_treshold) + "]" + "[" + str(
        segment_no_speech_prob) + ":" + str(segment_avg_logprob) + "]" + ".txt"
    original_text = "so we'll test some things. i think the first thing that we should maybe test is... no, i thought... no, no, no. so last one i think was the simulation of an actual session. i don't know if that's what you were trying. yeah, so what i think i'd like to try if that's okay with you is you speak. yeah. no, you can record. this is this is we want this information. okay, so essentially what we'll do is we'll have like so in the next like 10 minutes, we'll have a segment where maybe you just speak for like straight italian and for the next 10 minutes. a little bit of spanish, a little bit of english. then we'll have a section where we're both quiet for a little bit, for like maybe 30 seconds or a minute to test that. but i think we can do that all in the next 10 minutes. so if you'd like to begin, you know, maybe just say for the next 30 seconds, speak something in spanish and only spanish. yeah, or anything you would like that you can speak about for 30 seconds in spanish. sure. yeah. no, this is perfect. we can do it in this one. okay. okay, perfect. that's great. i think i understood a bit of that. we're going to do some of the sentences that we did. this is the second session. thank you for meeting with me today. thank you for meeting with me as well. okay, so perfect. i think that's good. and so let's try something if you're able to, if you're okay with it. let's repeat that same kind of agenda, but halfway through. let's maybe do like 15 seconds where you're speaking that same script in spanish. and then at the 15 second mark, let's move to you speaking english, if that's something that you're okay with. perfecto, gracias. okay, that's amazing. and then so i think i'm just going to talk. and so i think that can kind of like kill two kind of birds at one stone because we also want to test when it's super silent. so i think i'll speak for the next, say, minute i'll take a minute or two and in that time we'll also test silence on your end. since you'll be listening to me, your audio will be silent for the next two minutes. and then we can test that. i'm going to pull up a little script that i prepared. hello, my name is marcos abadi. i'm a fourth year computer science major at the university of san francisco. i am here both as a software developer and a language learner testing this model to see how well it performs with a variety of different scripts. i would really like to use this platform to get better at spanish and i'm excited to see what we can do. that was the english one. i will do the spanish one right now. and then hopefully... we will see what the transcription is. i'm excited to see what this model produces. yes. yeah, that could work. i think, yeah. now i will speak the following segment in spanish to contrast the model performance. hola, mi nombre es marcos abadi. soy un estudiante de 4to año en la universidad de san francisco. estoy aquí como un estudiante de informática y como un estudiante de lenguaje probando este modelo para ver que tal logra producir un texto a partir de diferentes idiomas. me encantaría usar esta plataforma para mejorar mi español. estoy muy emocionado de ver como funciona. you're good. yeah, that would be a good thing to test. we can see that. yeah, okay. yeah, i will read just the english and spanish. yeah. thank you so much for meeting with me today. gracias, buen dia profesor"
    with open(txt, "r") as f:
        text = f.read()
    text = text.strip()
    text = text.lower()
    text = text.replace("\n", " ")

    d = difflib.ndiff(original_text, text)
    deleted_text = ''.join([char[2] for char in d if char.startswith('- ')])

    return len(deleted_text)


def merge_intervals_with_k_treshold(intervals, k):
    merged = []
    for interval in intervals:
        if not merged:
            merged.append(interval)
        else:
            if merged[-1][1] + k > interval[0]:
                merged[-1][1] = interval[1]
            else:
                merged.append(interval)
    return merged


def concat_audio_files(files) -> str:
    with open("audio/concat.txt", "w") as f:
        for file in files:
            f.write("file '" + file + "'\n")

    sp.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", "audio/concat.txt",
         "-c", "copy",
         "output.mp4"])

    for file in files:
        os.remove("audio/" + file)

    return "output.mp4"


def extract_audio_segments(file, intervals) -> dict:
    files_and_timestamps = {}
    for segment in intervals:
        generated_uuid = str(uuid.uuid4())
        start = segment[0]
        end = segment[1]
        if start != end:
            sp.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", file, "-ss", str(start), "-to", str(end),
                 "-c",
                 "copy",
                 "audio/" + generated_uuid + ".mp4"])
            transcription_map[generated_uuid] = {
                "file": "audio/" + generated_uuid + ".mp4",
                "start": start,
                "end": end,
                "subject": "",
            }
        segment.append(generated_uuid)
    print(intervals)

    return files_and_timestamps


def remove_silence(file, stop_periods, stop_duration, stop_threshold):
    output = file[:5] + "silence.mp3"

    command = (f"ffmpeg -y -i {file} -af silenceremove="
               f"stop_periods={str(stop_periods)}:"
               f"stop_duration={str(stop_duration)}:"
               f"stop_threshold={str(stop_threshold)}dB "
               # f"start_duration=0.5:"
               # f"start_threshold=-42dB "
               f"{output}")

    sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT)

    return output


def detect_silence(file, noice_threshold="-50dB", duration=1):
    '''
    source: https://onkar-patil.medium.com/how-to-remove-silence-from-an-audio-using-python-50fd2c00557d
    This function is a python wrapper to run the ffmpeg command in python and extranct the desired output

    path= Audio file path
    time = silence time threshold

    returns = list of tuples with start and end point of silences
    '''
    command = "ffmpeg -i " + file + f" -af silencedetect=n={noice_threshold}:d=" + str(duration) + " -f null -"
    out = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT)
    stdout, stderr = out.communicate()
    s = stdout.decode("utf-8")
    k = s.split('[silencedetect @')
    if len(k) == 1:
        # print(stderr)
        return None

    start, end = [], []
    for i in range(1, len(k)):
        x = k[i].split(']')[1]
        if i % 2 == 0:
            x = x.split('|')[0]
            x = x.split(':')[1].strip()
            minutes = float(x) / 60
            seconds = 60 * (minutes - int(minutes))
            time = f"{int(minutes)}:{int(seconds)}"
            end.append(float(x))
        else:
            x = x.split(':')[1]
            x = x.split('size')[0]
            x = x.replace('\r', '')
            x = x.replace('\n', '').strip()
            minutes = float(x) / 60
            seconds = 60 * (minutes - int(minutes))
            time = f"{int(minutes)}:{int(seconds)}"
            start.append(float(x))
    # remove_silence(file, -1, 1, -40)

    intervals = list(zip(start, end))
    res = [list(ele) for ele in intervals]

    return res


def transcribe_whisper(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor],
        *,
        verbose: Optional[bool] = None,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        **decode_options,
):
    """
    Transcribe an audio file using Whisper with mutiple languages.

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successively used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    word_timestamps: bool
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.

    prepend_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the next word

    append_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the previous word

    initial_prompt: Optional[str]
        Optional text to provide as a prompt for the first window. This can be used to provide, or
        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
        to make it more likely to predict those word correctly.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    # print(f"Transcribing audio...")
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    # Pad 30-seconds of silence to the input audio, for slicing
    mel = log_mel_spectrogram(audio, padding=N_SAMPLES)
    content_frames = mel.shape[-1] - N_FRAMES

    if decode_options.get("languages", None) is None:
        if not model.is_multilingual:
            decode_options["languages"] = ["en"]
        else:
            if verbose:
                print(
                    "Detecting language using up to the first 30 seconds. Use `--language` to specify the language"
                )
            mel_segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)
            _, probs = model.detect_language(mel_segment)
            decode_options["languages"] = probs
            if verbose is not None:
                print(
                    f"Detected language: {LANGUAGES[decode_options['languages']].title()}"
                )

    languages: List[str] = decode_options["languages"]
    task: str = decode_options.get("task", "transcribe")
    tokenizers = {language: get_tokenizer(model.is_multilingual, language=language, task=task) for language in
                  languages}

    if word_timestamps and task == "translate":
        warnings.warn("Word-level timestamps on translations may not be reliable.")

    def decode_with_max(segment: torch.Tensor) -> DecodingResult:
        temperatures = (
            [temperature] if isinstance(temperature, (int, float)) else temperature
        )
        decode_result = None
        results = []

        for t in temperatures:
            if len(results) > 0:
                break  # already succeeded at previous temperature
            for language in languages:
                kwargs = {**decode_options}
                kwargs.pop("languages", None)
                kwargs.update({"language": language})
                if t > 0:
                    # disable beam_size and patience when t > 0
                    kwargs.pop("beam_size", None)
                    kwargs.pop("patience", None)
                else:
                    # disable best_of when t == 0
                    kwargs.pop("best_of", None)

                options = DecodingOptions(**kwargs, temperature=t)
                decode_result = model.decode(segment, options)

                needs_fallback = False
                if (
                        compression_ratio_threshold is not None
                        and decode_result.compression_ratio > compression_ratio_threshold
                ):
                    needs_fallback = True  # too repetitive
                if (
                        logprob_threshold is not None
                        and decode_result.avg_logprob < logprob_threshold
                ):
                    needs_fallback = True  # average log probability is too low
                if (
                        no_speech_threshold is not None
                        and decode_result.no_speech_prob > no_speech_threshold
                ):
                    needs_fallback = False  # silence
                if not needs_fallback:
                    results.append(decode_result)
            else:
                # no break
                results.append(decode_result)
                continue

        return min(results, key=lambda r: abs(r.avg_logprob))

    seek = 0
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    if initial_prompt is not None:
        initial_prompt_tokens = tokenizers[languages[0]].encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []

    def new_segment(
            *, start: float, end: float, tokens: torch.Tensor, result: DecodingResult
    ):
        tokens = tokens.tolist()
        text_tokens = [token for token in tokens if token < tokenizers[result.language].eot]
        return {
            "seek": seek,
            "start": start,
            "end": end,
            "text": tokenizers[result.language].decode(text_tokens),
            "tokens": tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        }

    # show the progress bar when verbose is False (if True, transcribed text will be printed)
    with tqdm.tqdm(
            total=content_frames, unit="frames", disable=verbose is not False
    ) as pbar:
        last_speech_timestamp = 0.0
        while seek < content_frames:
            time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            mel_segment = mel[:, seek: seek + N_FRAMES]
            segment_size = min(N_FRAMES, content_frames - seek)
            segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(model.device).to(dtype)

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result: DecodingResult = decode_with_max(mel_segment)
            tokens = torch.tensor(result.tokens)
            tokenizer = tokenizers[result.language]
            previous_language = None

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if (
                        logprob_threshold is not None
                        and result.avg_logprob > logprob_threshold
                ):
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment_size  # fast-forward to the next segment boundary
                    continue

            previous_seek = seek
            current_segments = []

            timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
            single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]

            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
            consecutive.add_(1)
            if len(consecutive) > 0:
                # if the output contains two consecutive timestamp tokens
                slices = consecutive.tolist()
                if single_timestamp_ending:
                    slices.append(len(tokens))

                last_slice = 0
                for current_slice in slices:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_pos = (
                            sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_pos = (
                            sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )
                    current_segments.append(
                        new_segment(
                            start=time_offset + start_timestamp_pos * time_precision,
                            end=time_offset + end_timestamp_pos * time_precision,
                            tokens=sliced_tokens,
                            result=result,
                        )
                    )
                    last_slice = current_slice

                if single_timestamp_ending:
                    # single timestamp at the end means no speech after the last timestamp.
                    seek += segment_size
                else:
                    # otherwise, ignore the unfinished segment and seek to the last timestamp
                    last_timestamp_pos = (
                            tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                    )
                    seek += last_timestamp_pos * input_stride
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if (
                        len(timestamps) > 0
                        and timestamps[-1].item() != tokenizer.timestamp_begin
                ):
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    last_timestamp_pos = (
                            timestamps[-1].item() - tokenizer.timestamp_begin
                    )
                    duration = last_timestamp_pos * time_precision

                current_segments.append(
                    new_segment(
                        start=time_offset,
                        end=time_offset + duration,
                        tokens=tokens,
                        result=result,
                    )
                )
                seek += segment_size

            if word_timestamps:
                add_word_timestamps(
                    segments=current_segments,
                    model=model,
                    tokenizer=tokenizer,
                    mel=mel_segment,
                    num_frames=segment_size,
                    prepend_punctuations=prepend_punctuations,
                    append_punctuations=append_punctuations,
                    last_speech_timestamp=last_speech_timestamp,
                )
                word_end_timestamps = [
                    w["end"] for s in current_segments for w in s["words"]
                ]
                if len(word_end_timestamps) > 0:
                    last_speech_timestamp = word_end_timestamps[-1]
                if not single_timestamp_ending and len(word_end_timestamps) > 0:
                    seek_shift = round(
                        (word_end_timestamps[-1] - time_offset) * FRAMES_PER_SECOND
                    )
                    if seek_shift > 0:
                        seek = previous_seek + seek_shift

            if verbose:
                for segment in current_segments:
                    start, end, text, lang, avglog = segment["start"], segment["end"], segment["text"], result.language, \
                        segment["avg_logprob"]
                    line = f"{lang}-{avglog}[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
                    print(make_safe(line))

            # if a segment is instantaneous or does not contain text, clear it
            for i, segment in enumerate(current_segments):
                if segment["start"] == segment["end"] or segment["text"].strip() == "":
                    segment["text"] = ""
                    segment["tokens"] = []
                    segment["words"] = []

            all_segments.extend(
                [
                    {"id": i, **segment}
                    for i, segment in enumerate(
                    current_segments, start=len(all_segments)
                )
                ]
            )
            all_tokens.extend(
                [token for segment in current_segments for token in segment["tokens"]]
            )

            if not condition_on_previous_text or result.temperature > 0.5 or result.language != previous_language:
                # do not feed the prompt tokens if a high temperature was used or the language changed
                prompt_reset_since = len(all_tokens)
            previous_language = result.language

            # update progress bar
            pbar.update(min(content_frames, seek) - previous_seek)

    return dict(
        text="\n".join([segment["text"] for segment in all_segments]),
        segments=all_segments,
        languages=languages,
    )


def write_text_to_file(text, filename):
    output = "results/" + filename + "text" + ".txt"
    with open(output, "w") as f:
        f.write(text)


def get_non_silent_parts(intervals):
    non_silent_parts = []
    for i in range(len(intervals) - 1):
        start = intervals[i][1]
        end = intervals[i + 1][0]
        if end - start > 0.018:
            non_silent_parts.append([intervals[i][1], intervals[i + 1][0]])

    return non_silent_parts


torch.cuda.init()
device = "cuda"
model = whisper.load_model("small").to(device)
# files = ["silence_vale.mp3"]
files = ["frank_guglielmo_mock_session.mp4", "valerio_mock_session.mp4"]
# files = ["meetisilence.mp3"]
# files = ["franksilence.mp3", "valersilence.mp3"]
for file in files:
    silence_parts = detect_silence(file, "-70dB", 2.5)
    non_silent_parts = merge_intervals_with_k_treshold(get_non_silent_parts(silence_parts), 5)
    extract_audio_segments(file, non_silent_parts)

for key, value in transcription_map.items():
    languages = ["en", "it"]
    transcription = transcribe_whisper(model, value["file"], verbose=None, languages=languages, word_timestamps=True)
    transcription_map[key]["transcription"] = "[teacher]: " + transcription["text"]

sorted_items = sorted(transcription_map.items(), key=lambda x: x[1]["start"])
for key, value in sorted_items:
    print(value["transcription"])

# print(detect_silence(file, "-50dB", 1))
# silence_parts = detect_silence(file, "-55dB", 1)
# intervals = merge_intervals_with_k_treshold(silence_parts, 0.5)
# print(get_non_silent_parts(silence_parts))
# remove_silence(file, -1, 1, -40)
# languages = ["en", "it"]
#
# result = transcribe_whisper(model, file, verbose=False, languages=languages, word_timestamps=True)
# write_text_to_file(result["text"], file)

# transcribe_whisper(model, file, verbose=True, languages=languages, word_timestamps=True)
# error_bucket = transcribe(model, file, "en", .6, -2, 0.4, -0.7)
# print(error_bucket)
# intervals = merge_intervals_with_k_treshold(error_bucket, 5)
# print(intervals)
# output_file = extract_audio_segments(file, intervals)
# transcribe(model, output_file, "it", 0.6, -1, 0.35, -0.7)
# remove_silence(file, -1, 0.1, -50)
# print(detect_silence(file, "-50db", 1))

# Delete the files in the results directory
# for file in os.listdir("results/"):
#     os.remove("results/" + file)
