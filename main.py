# to do:
# add audio preprocessing
# add a module that would enable data preprocessing if necessary
# add a module to split the audio track
from vosk import Model, KaldiRecognizer
import json
import wave
from moviepy.editor import VideoFileClip

# separating the sound from the video
audio_f = VideoFileClip("video_1.mp4").audio
# saving the audio file
audio_f.write_audiofile("output.wav")
# initializing a model with a small language pack
model = Model(r"vosk-model-small-ru-0.4")

wf = wave.open(r'output.wav', "rb")
rec = KaldiRecognizer(model, 8000)

result = ''
last_n = False

while True:
    # the frequency is taken from the documentation,
    # but you can try to find better values
    data = wf.readframes(8000)

    if len(data) == 0:
        break

    if rec.AcceptWaveform(data):
        # word recognition
        res = json.loads(rec.Result())
        # adding words to the result variable
        if res['text'] != '':
            result += f" {res['text']}"
            last_n = False
        # if there were no words, then an empty string is added
        elif not last_n:
            result += '\n'
            last_n = True

res = json.loads(rec.FinalResult())
result += f" {res['text']}"

print(result)
