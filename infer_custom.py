import librosa
import numpy as np
import SpeechModels
from tensorflow.keras.models import Model
from argparse import ArgumentParser
import time
from utils import *

nCategs = 10
sr = 16000
seconds = 1

classes = ['sai', 'sang trái', 'sang phải', 'tiến lên',\
           'lùi xuống', 'không', 'có', 'khởi động', 'tắt nguồn']

model = SpeechModels.AttRNNSpeechModel(nCategs, samplingrate = sr, inputLength = None)#, rnn_func=L.LSTM)
# model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
model.load_weights('model-attRNN_vn9.h5')

attSpeechModel = Model(inputs=model.input,
                                 outputs=[model.get_layer('output').output, 
                                          model.get_layer('attSoftmax').output,
                                          model.get_layer('mel_stft').output])
def pad(data, dim=16000):
    if data.shape[0] == dim:
        return data
    elif data.shape[0] > dim:  # bigger
        # we can choose any position in curX-self.dim
        randPos = np.random.randint(data.shape[0] - dim)
        return data[randPos:randPos + dim]
    else:  # smaller
        randPos = np.random.randint(dim - data.shape[0])
        X = np.empty((1, dim))
        X[0, randPos:randPos + data.shape[0]] = data
        return X

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--record", default=None)
    parser.add_argument("--audio", default=None)

    opt = parser.parse_args()

    if opt.record == None:
        audio, sr = librosa.load(opt.audio, sr=None)
        audio = pad(audio, 16000)
        audio = audio.reshape(1,16000)
        outs, attW, specs = attSpeechModel.predict(audio)
        print(classes[np.argmax(outs, axis=1)[0] - 1])
    else:
        while (1):
            _ = record_(seconds)
            audio, sr = librosa.load('output.wav', sr=16000, mono=True)
            audio = pad(audio, 16000)
            audio = audio.reshape(1,16000)
            outs, attW, specs = attSpeechModel.predict(audio)
            print(np.max(outs,axis=1))
            print(classes[np.argmax(outs,axis=1)[0]-1])
            time.sleep(3)

