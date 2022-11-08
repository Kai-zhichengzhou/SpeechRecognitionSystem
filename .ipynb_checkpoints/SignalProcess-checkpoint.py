from pydub import AudioSegment
import IPython.display as Ipd
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

FRAME_SIZE = 2048
HOP_SIZE = 512

#Define the convertion function from compressed format to wav for data
def ConvertionToWav(file_dir, file_dest):
    #file_dir - original mp3 data location
    #file_dest - the location to write wav format data

    data_dir = os.walk(file_dir)
    for path, dir_name, file_list in data_dir:

        for voice_sample in file_list:
            sample_path = os.path.join(path, voice_sample)
            output_path = file_dest + voice_sample.split('.')[0] + '.wav' #necessray to match the name
            audSeg = AudioSegment.from_mp3(sample_path)
            audSeg.export(output_path, format = 'wav')


def getMelspectrogram(waveform, n_fft = FRAME_SIZE, hop_length = HOP_SIZE, n_mels = 128, fmin = 10,fmax = 8000):


    #extract melspectrogram using params passed in 
    mel_spec = librosa.feature.melspectrogram(waveform,n_fft = n_fft, 
                        hop_length= hop_length, n_mels = n_mels, fmin = fmin, fmax = fmax)

    #librosa extract melspectrogram as sqaures magnitude of spectrogram
    #we need melspectrogram in decibels

    mel_spec_db = librosa.power_to_db(mel_spec)

    return mel_spec_db
# def getMelspectrogram(file_path, n_fft = FRAME_SIZE, hop_length = HOP_SIZE, n_mels = 128, fmin = 10,fmax = 8000):

#     waveform, sr = librosa.load(file_path)

#     #extract melspectrogram using params passed in 
#     mel_spec = librosa.feature.melspectrogram(waveform, sr = sr, n_fft = n_fft, 
#                         hop_length= hop_length, n_mels = n_mels, fmin = fmin, fmax = fmax)

#     #librosa extract melspectrogram as sqaures magnitude of spectrogram
#     #we need melspectrogram in decibels

#     mel_spec_db = librosa.power_to_db(mel_spec)

#     return mel_spec_db


def getComprehensiveMFCCs(file_path, n_mfcc = 13):

    #calcaluate the comprehensive mfccs (mfcc, delta_mfcc, delta2_mfcc)
    
    waveform, sr = librosa.load(file_path)
    print(type(waveform))
    mfccs = librosa.feature.mfcc(waveform, n_mfcc = 13)

    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order = 2)

    comprehensive_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

    return comprehensive_mfccs



