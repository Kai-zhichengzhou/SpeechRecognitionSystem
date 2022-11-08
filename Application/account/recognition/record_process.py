import wave
import pyaudio
import torch
import time
import torchaudio
import torchaudio.transforms as transform
import torch.nn as nn
import sys


sys.path.append("..")

sample_rate = 16000
n_mels = 128
n_fft = 350
frame_size = 350
hop_size = frame_size // 2

class Recorder():

    def __init__(self):
        self.recording = True
        self.seconds = 6

    

            
    def record(self):

        audio = pyaudio.PyAudio()
        stream = audio.open(format = pyaudio.paInt16, channels = 1, rate = 16000,input = True, frames_per_buffer = 1024)
        frames = []
        # try:
        end_time = time.time() + self.seconds
        while time.time() < end_time:
            data = stream.read(1024)
            frames.append(data)


        # except KeyboardInterrupt:
        #     pass

        # i = 0
        # while i < 300:
        #         data = stream.read(1024)
        #         frames.append(data)
        #         i += 1
        print("debug here")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        file_name = "record_2.wav"
        rec_file = wave.open(file_name, "wb")
        rec_file.setnchannels(1)
        rec_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        rec_file.setframerate(16000)
        rec_file.writeframes(b''.join(frames))
        rec_file.close()
        print("success record")
        return file_name

    
# def record():

#     audio = pyaudio.PyAudio()
#     stream = audio.open(format = pyaudio.paInt16, channels = 1, rate = 16000,input = True, frames_per_buffer = 1024)

#     frames = []
    
#     while True:
#         data = stream.read(1024)
#         frames.append(data)



#     # i = 0
#     # while i < 300:
#     #         data = stream.read(1024)
#     #         frames.append(data)
#     #         i += 1

#     stream.stop_stream()
#     stream.close()
#     audio.terminate()

#     file_name = "record_2.wav"

#     rec_file = wave.open(file_name, "wb")
#     rec_file.setnchannels(1)
#     rec_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
#     rec_file.setframerate(16000)

#     rec_file.writeframes(b''.join(frames))
#     rec_file.close()
#     print("success record")
#     return file_name


def get_MelSpectrogram(sample_rate, n_fft, n_mels, frame_size, hop_size, f_min = 0):
    melspec_transform = transform.MelSpectrogram(sample_rate= sample_rate, n_mels=n_mels)

    return melspec_transform

def voice_preprocess(file):




    waveform_1, _ = torchaudio.load(file)

    #melspec_transform = transform.MelSpectrogram(sample_rate, n_fft, n_mels,frame_size, hop_size)
    melspec_transform = transform.MelSpectrogram()
    melspec = melspec_transform(waveform_1)
    melspec = melspec.squeeze(0) #(channel, n_mels, time) -> time, n_mels

    return melspec

def main():

    # newFile = record()
    recorder = Recorder()
    newFile = recorder.record()
    return voice_preprocess(newFile)








