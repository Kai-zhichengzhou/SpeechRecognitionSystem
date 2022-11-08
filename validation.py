import os
# from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np

import mapping 
import dataPreprocessing.preprocessing as preprocess
import Decoder.decode_algorithm as Decoder


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(device)

text_transform = mapping.TextMapping()

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

def valid_data_processing(data):
    Mel_spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, label, _, _, _) in data:

        MelSpec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        #print("labels::", label)
        Mel_spectrograms.append(MelSpec)
        a = text_transform.convert_TextToInt(label)
        b = text_transform.convert_IntToText(a)
       # print("labels::", label)
        #print("decode label:", b)
        label = torch.Tensor(text_transform.convert_TextToInt(label))
        labels.append(label)

        input_length = MelSpec.shape[0] // 2
        input_lengths.append(input_length)
        label_length = len(label)
        label_lengths.append(label_length)

    MelSpectrograms = nn.utils.rnn.pad_sequence(Mel_spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return MelSpectrograms, labels, input_lengths, label_lengths

class LayerNormalization(nn.Module):
    '''
    A class that normalize the input value along all feature-dimensions

    Attributes
    ----------
    self.norm: torch.nn.module
        a module called from torch.nn to implement layer nomralization with certain nunber of features

    Methods
    ------
    forward(self, x)
        apply forward compuation to implement normalizations on all features

    '''
    
    def __init__(self, num_features):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(num_features)
        
    def forward(self, x):
        x = x.transpose(2,3).contiguous()
        x = self.norm(x)
        return x.transpose(2,3).contiguous()
    


#Define Resnet and LSTM Model 
class ResNet(nn.Module):
    '''
    A class that designs Residual Neural Network Block of CNN-architecture with convolution layer, 
    normalization layer and skip connection at the end of the block


    Attributes
    ----------
    self.conv_1: torch.nn.module
        a module called from torch.nn to implement convolution computation
    self.conv_2: torch.nn.module
        a module called from torch.nn to implement convolution computation
    self.layerNorm_1: LayerNomrlization
        A LayerNomrlization Object created to apply normalization on input data
    self.layerNorm_2: LayerNomrlization
        A LayerNomrlization Object created to apply normalization on input data
    self.dropout_1: torch.nn.module
        A module called from torch.nn to implement Dropout with dropout rate to reduce overfitting
    self.dropout_1: torch.nn.module
        A module called from torch.nn to implement Dropout with dropout rate to reduce overfitting

    Methods
    ------
    forward(self, x)
        apply forward compuation in the block of defined residual neural network. 
        It involves normalizing the features, using Gelu activation to get activation value, apply dropout to recude
        overfitting, and apply convolutions for feature capturing. The residual is added at the end to avoid learning 
        degeneration.
    '''
        
    
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
        x = self.conv_1(x)
        
        x = self.layerNorm_2(x)
        x = F.gelu(x)
        x = self.dropout_2(x)
        x = self.conv_2(x)
        
        #concat the residual to the output for skip connection
        output = x + residual
        return output
    
        
    
    
class LSTM(nn.Module):

    '''
    A class that designs Long Short-term Memory block to resolve sequence-to-sequence task and compute value in each timestep


    Attributes
    ----------
    self.input_size: int
        A numner indicating the input size of the input sequence data
    self.hidden_size: int
        A numner indicating the number of neurons in hidden layer
    self.num_layers: int
        A number indicationg the layers of LSTM, or how many layers stacking
    self.lstm: torch.nn.module
        A module called from torch.nn to implement lstm network in Pytorch
    self.norm: LayerNomrlization
        A LayerNomrlization Object created to apply normalization on input data
    self.dropout torch.nn.module
        A module called from torch.nn to implement Dropout with dropout rate to reduce overfitting

    Methods
    ------
    forward(self, x)
        apply forward compuation in the block of defined LSTM network

    '''
    
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
    '''
    A class designed to aggregate all sub modules, e.g Residual Neural Network , Layernormalizion layer, LSTM to 
    build the whole model for automatic speech recognition task


    Attributes
    ----------
    self.num_features: int
        a module called from torch.nn to implement convolution computation
    self.conv_1: torch.nn.module
        a module called from torch.nn to implement convolution computation
    self.residual_nets: torch.nn.Sequential
        Aggregates a certain number of ResNet Objects to form a sequential pipeline 
    self.fc: torch.nn.module
        A module called from torch.nn to implement a fully-connect linear layer
    self.lstm_nets: torch.nn.Sequential
        Aggregates a certain number of LSTM Objects to form a sequential pipeline 
    self.lstm_to_linear: torch.nn.module
        A module called from torch.nn to implement a fully-connect linear layer from output of LSTM 
    self.classifer: torch.nn.module
        A module called from torch.nn to implement a fully-connect linear layer from input_size of dimensions to num_class dimensions


    Methods
    ------
    forward(self, x)
        apply forward compuation of the whole ASR model, including convolutions, ResNet block, fully-connect layer,
        LSTM and a linear classifer at the end. Activation of Gelu and dropout is applied during computation.
    '''
    
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


def evaluate(model, device, validLoader, criterion):
    '''
    The evalution function to validate the task with certain model and other hyperparamters.The results shows the 
    performance on unseen validation dataset to prevent overfitting and better hyperparameter-tuning


    Parameters
    ----------
    model: ASRModel
        The instance object of ASRModel for validation
    device: torch.device
        The status to indicate if the training is deployed on CPU or GPU
    validloader: torch.utils.data.dataloader
        The dataloader object to load,enumerate, and shuffle the input data 
    criterion: torch.nn.CTCLoss
        The module called from torch.nn.CTCLoss to implement the CTC Loss function in Pytorch


    '''
    model.eval()
    total_loss = 0
    wer, cer = [],[]
    speech_decoder = Decoder.SpeechDecoder()
    with torch.no_grad():
        for i, _data in enumerate(validLoader):
            melspectrograms, labels, input_lengths, label_lengths = _data 
            melspectrograms, labels = melspectrograms.to(device), labels.to(device)

            output = model(melspectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            total_loss += loss.item() / len(validLoader)

            decoded_preds, decoded_targets = speech_decoder.decode_text(output.transpose(0, 1), labels, label_lengths)
            #print("pred",decoded_preds)
            #print("label", decoded_targets)

            #calculate wer in this batch
            for j in range(len(decoded_preds)):
                wer.append(Decoder.calc_char_error_rate(decoded_targets[j], decoded_preds[j]))
                cer.append(Decoder.calc_word_error_rate(decoded_targets[j], decoded_preds[j]))


    average_cer = sum(cer)/len(cer)
    average_wer = sum(wer)/len(wer)

    print('Validation: loss: {:.4f}\n'.format(total_loss))
    with open('validation_2nd.txt', 'a') as f:
        f.write(f"EPOCH : {1} | loss : {total_loss} | WER: {average_wer} | CER: {average_cer}")



# learning_rate = 0.005
batch_size = 12
valid_url = "test-clean"
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

if not os.path.isdir("./data"):
    os.makedirs("./data")
valid_dataset = torchaudio.datasets.LIBRISPEECH("../Data", url=valid_url, download=True)

valid_dataset = torchaudio.datasets.LIBRISPEECH("../Data", url=valid_url, download=True)
        

valid_loader = data.DataLoader(dataset=valid_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=lambda x: valid_data_processing(x),num_workers = 8)

#load the trained data for validation
ASR_model= torch.load('./model_1/model_20_500.pth')
ASR_model.to(device)
criterion = nn.CTCLoss(blank=28).to(device)


for idx in range(11, 21):
    ASR_model = torch.load(f'./model_retrain_2/retrained_model__{idx}.pth')
    ASR_model.to(device)
    criterion = nn.CTCLoss(blank=28).to(device)
    evaluate(ASR_model, device, valid_loader, criterion)

