import librosa
import numpy as np
import SpeechModels
from tensorflow.keras.models import Model
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

model = SpeechModels.AttRNNSpeechModel(nCategs, samplingrate = sr, inputLength = None)#, rnn_func=L.LSTM)
model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
model.load_weights('models/model-attRNN.h5')

attSpeechModel = Model(inputs=model.input,
                                 outputs=[model.get_layer('output').output, 
                                          model.get_layer('attSoftmax').output,
                                          model.get_layer('mel_stft').output])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--record", default=None)
    parser.add_argument("--audio", default=None)

    opt = parser.parse_args()

    if opt.record == None:
        audio, sr = librosa.load(opt.audio, sr=None)
        audio = audio[:16000].reshape(1,16000)
        outs, attW, specs = attSpeechModel.predict(audio)
        print(classes[np.argmax(outs, axis=1)[0] - 1])
    else:
        while (1):
            _ = record_(seconds)
            audio, sr = librosa.load('output.wav', sr=16000, mono=True)
            audio = audio.reshape(1,16000)
            outs, attW, specs = attSpeechModel.predict(audio)
            print(np.max(outs,axis=1))
            print(classes[np.argmax(outs,axis=1)[0]-1])
            time.sleep(3)