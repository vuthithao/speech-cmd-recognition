"""
File containing scripts to download audio from various datasets

Also has tools to convert audio into numpy
"""
import os
import pandas as pd

import audioUtils


def PrepareSpeechCmd(task='vietnamese22'):
    """
    Prepares Google Speech commands dataset version 2 for use

    tasks:

    Returns full path to training, validation and test file list and file categories
    """
    basePath = 'data'

    if task == 'vietnamese9':
        GSCmdV2Categs = {
            'unknown': 0,
            'silence': 0,
            '_unknown_': 0,
            '_silence_': 0,
            '_background_noise_': 0,
            'sai': 1,
            'sang trái':2,
            'sang phải':3,
            'tiến lên':4,
            'lùi xuống':5,
            'không':6,
            'có':7,
            'khởi động':8,
            'tắt nguồn':9}
        numGSCmdV2Categs = 10

    # print('Converting test set WAVs to numpy files')
    # audioUtils.WAV2Numpy(basePath + '/test/')
    print('Converting training set WAVs to numpy files')
    audioUtils.WAV2Numpy(basePath + '/train/')

    # read split from files and all files in folders
    testWAVs = pd.read_csv(basePath + '/train/testing_list.txt',
                           sep=" ", header=None)[0].tolist()
    valWAVs = pd.read_csv(basePath + '/train/validation_list.txt',
                          sep=" ", header=None)[0].tolist()

    testWAVs = [os.path.join(basePath + '/train/', f + '.npy')
                for f in testWAVs if f.endswith('.wav')]
    valWAVs = [os.path.join(basePath + '/train/', f + '.npy')
               for f in valWAVs if f.endswith('.wav')]
    allWAVs = []
    for root, dirs, files in os.walk(basePath + '/train/'):
        allWAVs += [root + '/' + f for f in files if f.endswith('.wav.npy')]
    trainWAVs = list(set(allWAVs) - set(valWAVs) - set(testWAVs))

    # testWAVsREAL = []
    # for root, dirs, files in os.walk(basePath + '/test/'):
    #     testWAVsREAL += [root + '/' +
    #                      f for f in files if f.endswith('.wav.npy')]

    # get categories
    testWAVlabels = [_getFileCategory(f, GSCmdV2Categs) for f in testWAVs]
    valWAVlabels = [_getFileCategory(f, GSCmdV2Categs) for f in valWAVs]
    trainWAVlabels = [_getFileCategory(f, GSCmdV2Categs) for f in trainWAVs]
    # testWAVREALlabels = [_getFileCategory(f, GSCmdV2Categs)
    #                      for f in testWAVsREAL]

    # # background noise should be used for validation as well
    # backNoiseFiles = [trainWAVs[i] for i in range(len(trainWAVlabels))
    #                   if trainWAVlabels[i] == GSCmdV2Categs['silence']]
    # backNoiseCats = [GSCmdV2Categs['silence']
    #                  for i in range(len(backNoiseFiles))]
    # if numGSCmdV2Categs == 12:
    #     valWAVs += backNoiseFiles
    #     valWAVlabels += backNoiseCats

    # build dictionaries
    testWAVlabelsDict = dict(zip(testWAVs, testWAVlabels))
    valWAVlabelsDict = dict(zip(valWAVs, valWAVlabels))
    trainWAVlabelsDict = dict(zip(trainWAVs, trainWAVlabels))
    # testWAVREALlabelsDict = dict(zip(testWAVsREAL, testWAVREALlabels))

    # a tweak here: we will heavily underuse silence samples because there are few files.
    # we can add them to the training list to reuse them multiple times
    # note that since we already added the files to the label dicts we don't
    # need to do it again

    # for i in range(200):
    #     trainWAVs = trainWAVs + backNoiseFiles

    # info dictionary
    trainInfo = {'files': trainWAVs, 'labels': trainWAVlabelsDict}
    valInfo = {'files': valWAVs, 'labels': valWAVlabelsDict}
    testInfo = {'files': testWAVs, 'labels': testWAVlabelsDict}
    # testREALInfo = {'files': testWAVsREAL, 'labels': testWAVREALlabelsDict}
    gscInfo = {'train': trainInfo,
               'test': testInfo,
               'val': valInfo}
               # 'testREAL': testREALInfo}

    print('Done preparing Speech commands dataset')

    return gscInfo, numGSCmdV2Categs


def _getFileCategory(file, catDict):
    """
    Receives a file with name sd_GSCmdV2/train/<cat>/<filename> and returns an integer that is catDict[cat]
    """
    categ = os.path.basename(os.path.dirname(file))
    return catDict.get(categ, 0)

