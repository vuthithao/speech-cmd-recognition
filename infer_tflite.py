import librosa
import numpy as np
from argparse import ArgumentParser
import time
from utils import *

nCategs = 36
sr = 16000
seconds = 1
classes = ['nine', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
           'zero', 'one', 'two', 'three', 'four', 'five', 'six',
           'seven',  'eight', 'backward', 'bed', 'bird', 'cat', 'dog',
           'follow', 'forward', 'happy', 'house', 'learn', 'marvin', 'sheila', 'tree',
           'visual', 'wow']

# load tfl model
tfl_file = "models/cmdRecognition.tflite"
model = load_tflite_model(tfl_file)
print('Loaded Model')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--record", default=None)
    parser.add_argument("--audio", default=None)

    opt = parser.parse_args()

    if opt.record == None:
        audio, sr = librosa.load(opt.audio, sr=None)
        audio = audio[:16000].reshape(1,16000)
        outs = predict(model, audio)
        print(classes[np.argmax(outs, axis=1)[0] - 1])
    else:
        while (1):
            _ = record_(seconds)
            audio, sr = librosa.load('output.wav', sr=16000, mono=True)
            audio = audio.reshape(1,16000)
            outs = predict(model, audio)
            print(np.max(outs,axis=1))
            print(classes[np.argmax(outs,axis=1)[0]-1])
            time.sleep(3)