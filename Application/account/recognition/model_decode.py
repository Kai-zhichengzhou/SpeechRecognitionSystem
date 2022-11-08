from typing import Text

import torch
from .mapping import TextMapping
import torchaudio.transforms as transform
import torch.nn as nn
import torch.nn.functional as F
import sys
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

    '''
    A class to implement a voice recorder  - a recorder that uses I/O to record user's voice and save the speech
    into local directory as a waveform (WAV) format 

    Attributes
    ----------
    self.recording: int
        A boolean value to indicate if it is currently recording
    self.recording_seconds: int
        A number to set up the recording duration

    Methods
    -------
    record(self)
        The function to initilize a stream for input recording and record the voice with certain hyperparamters 
        and save the speech recording into local directory


    '''

    def __init__(self):
        self.recording = True
        self.recording_seconds = 10

    
     
    def record(self):
        '''
        The function to initilize a stream for input recording and record the voice with certain hyperparamters 
        and save the speech recording into local directory

        Return
        ------
        string
            The function returns the file name of recorded file
        '''

        audio = pyaudio.PyAudio()
        stream = audio.open(format = pyaudio.paInt16, channels = 1, rate = 16000,input = True, frames_per_buffer = 1024)
        frames = []
        # try:
        end_time = time.time() + self.recording_seconds
        while time.time() < end_time:
            data = stream.read(1024)
            frames.append(data)

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
        print("successfully record")
        return file_name




def get_MelSpectrogram(sample_rate, n_fft, n_mels, frame_size, hop_size):
    '''
    The function that takes in the raw input data and process the raw data to be suitable for later training.The 
    feature engineering and audio augmentation are applied inside the function

    Parameters
    ----------
    data_raw: Numpy.ndarray
        The ndarray stands for the input data that load from the dataset

    Return
    ------
    Numpy.ndarray, Numpy.ndarray, list(int), list(int)
        The function returns the mel-spectrogram after preprocessing pipeline, ground truth label, and lengths of both.

    '''
    melspec_transform = transform.MelSpectrogram(sample_rate= sample_rate, n_mels=n_mels)

    return melspec_transform


def voice_preprocess(file):
    '''
    The function implements the pipeline of preprocessing the recorded voice from user 

    Parameters
    ----------
    file: string
        The string of the recorded file name 

    Return
    ------
    Numpy.ndarray
        The function returns the mel-spectrogram processed and extracted from the recorded file

    '''


    waveform_1, _ = torchaudio.load(file)
    melspec_transform = transform.MelSpectrogram()
    melspec = melspec_transform(waveform_1)
    melspec = melspec.squeeze(0) #(channel, n_mels, time) -> time, n_mels
    return melspec

def load_and_process():
    '''
    Create an instance of recorder and record the voice from the user

    Return
    ------
    Numpy.ndarray
        The function creates the object of Recorder, record the voice from user, and returns the features of the input data
    '''

    recorder = Recorder()
    newFile = recorder.record()
    return voice_preprocess(newFile)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#create a object of textmapping for conversion between index and chars
text_mapping = TextMapping()


class LayerNormalization(nn.Module):
    
    def __init__(self, num_features):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(num_features)
        
    def forward(self, x):
        x = x.transpose(2,3).contiguous()
        x = self.norm(x)
        return x.transpose(2,3).contiguous()
    


#Define Resnet and LSTM Model 
class ResNet(nn.Module):
    
    def __init__(self, input_channel, output_channel, kernel, stride, num_features,dropout):
        
        super(ResNet,self).__init__()
        
        self.conv_1 = nn.Conv2d(input_channel, output_channel, kernel, stride, padding = kernel // 2)
        self.conv_2 = nn.Conv2d(output_channel, output_channel, kernel, stride, padding = kernel // 2)
        #apply layer normolization to input
        self.layerNorm_1 = LayerNormalization(num_features)
        self.layerNorm_2 = LayerNormalization(num_features)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x):
        
        residual = x 
        x = self.layerNorm_1(x)
        x = F.gelu(x) # perform better than Relu
        x = self.dropout_1(x)
#         print(x.shape)
#         print(x)
        x = self.conv_1(x)
        
        x = self.layerNorm_2(x)
        x = F.gelu(x)
        x = self.dropout_2(x)
        x = self.conv_2(x)
        
        #concat the residual to the output for skip connection
        output = x + residual
        return output
    
        
    
    
class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes = 29, batch_first = True):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, 
                           bidirectional = True)
        
        self.norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):

        
        

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        x = self.norm(x)

        x = F.gelu(x)
        
        output, _ = self.lstm(x, (h0, c0))
        output = self.dropout(output)

        
        return output


#Build the whole ASR model
    
class ASRModel(nn.Module):
    
    def __init__(self, num_resnet, num_lstm, input_size,num_class, num_features, stride, dropout = 0.1):
        super(ASRModel, self).__init__()
        num_features = num_features // 2
        
        self.conv_1 = nn.Conv2d(1, 32, 3, stride = stride, padding = 1)
        
        #apply residual nets 
        self.residual_nets = nn.Sequential(*[ResNet(32, 32, kernel = 3, 
                                                   stride = 1, dropout = dropout, num_features = num_features)
                                             for num in range(num_resnet)
        ])
        self.fc = nn.Linear(num_features * 32, input_size)
        
        self.lstm_nets = nn.Sequential(*[LSTM(input_size = input_size if num == 0 else input_size * 2 , hidden_size = input_size,num_layers =1, batch_first = num == 0,dropout = dropout) for num in range(num_lstm)
        ])
        
        self.activation = nn.GELU()
        self.Dropout = nn.Dropout(dropout)
        self.lstm_to_linear = nn.Linear(input_size * 2, input_size)
        self.classifier = nn.Linear(input_size, num_class)
        
    def forward(self, x):
        

        x = self.conv_1(x)
        x = self.residual_nets(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # transpose to (batch, time, feature) for lstm 
        x = self.fc(x)

        x = self.lstm_nets(x)
        x = self.lstm_to_linear(x)
        x = self.activation(x)
        x = self.Dropout(x)
        x = self.classifier(x)
        
        return x


class SpeechDecoder():
    '''
    A class to implement the decoding algorithm to translate output probability matrix compuated by the trained 
    model to decoded texts

    Attributes
    ----------
    self.blank_label: int
        A number, or index that stands for the blank label, or epilson during decoding
    self.repeat_collapse: boolean
        A boolean value to indicate that if collaspe the repetition

    Methods
    -------
    decode_text(self, output_matrix)
        The function takes in the output probability matrix computed by the model and input data, then 
        choose the character with largest computed probability in each time step and return the decoded
        text based on decoding rules

    Decode(self, ASR_model)
        The function load the model from parameter and call data preprocessing function to achieve processed features.
        At the final step the function calls decode_text function, which is from same class, and apply the model and input 
        to return the decodes the texts


    '''

    def __init__(self):
        self.blank_label = 28
        self.repeat_collaspse = True

    def decode_text(self, output_matrix):

        texts = []
        
        sequence_argmax = torch.argmax(output_matrix, dim = 2)

        for _, seq in enumerate(sequence_argmax):
            
            tmp = []

            for i, index in enumerate(seq):
                #collapse repetition and skip blank
                
                if index == self.blank_label:
                    continue
                if self.repeat_collaspse and i != 0 and index == seq[i - 1]:
                    continue
                tmp.append(index.item())

            texts.append(text_mapping.convert_IntToText(tmp))

        return texts


    def Decode(self, ASR_model):
        
        ASR_model.eval()
        melspec = load_and_process()
        # melspec = record_process.main()
        melspec = melspec.unsqueeze(0).unsqueeze(1)

        output = ASR_model(melspec)
        output = F.log_softmax(output, dim = 2)
        text_decode = self.decode_text(output)[0]
        print("Speech Recognition is working here")

        return text_decode










