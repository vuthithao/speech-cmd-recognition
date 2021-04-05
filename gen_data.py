import requests
import subprocess
import numpy as np
import os
from scipy.io import wavfile
from tqdm import tqdm

def resample_single(wav_from, wav_to):
    subprocess.call(['sox', wav_from, "-r", "16000", wav_to])
    return 0

def float64_to_int16(dat):
    data = dat.copy()
    # data = data.astype(np.int64)
    max_val = max(np.abs(data))
    data = dat/max_val
    data *= 32767
    data = data.astype(np.int16)

    return data

def gen(file, text, alpha):
    # Call text to speech API
    f = 'temp.wav'
    r = requests.post("http://103.137.4.6:3333/forward/to-speech", json={'alpha': alpha, 'text': r'{}'.format(text)})
    res = r.json()
    audio = res['audio']
    audio16 = float64_to_int16(audio)
    wavfile.write(file, 22050, np.array(audio16))
    # resample_single(f, file)

command = ['sai', 'sang trái', 'sang phải', 'tiến lên',\
           'lùi xuống', 'không', 'có', 'khởi động', 'tắt nguồn']


alphas = [round(i,1) for i in list(np.arange(0.8, 1.2, 0.1))]

if os.path.exists('dataset') == False:
    os.mkdir('dataset')
for i in command:
    print(i)
    os.mkdir('dataset/' + i)
    for j in tqdm(alphas):
        file = 'dataset/{}/bm-fm_'.format(i) + str(j) + '.wav'
        gen(file, i, j)
