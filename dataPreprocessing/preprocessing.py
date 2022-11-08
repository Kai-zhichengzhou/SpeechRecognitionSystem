from cProfile import label
import torch
import torchaudio.transforms as transform
import torch.nn as nn
from ..mapping import TextMapping


sample_rate = 16000
n_mels = 128
#frame_size = 450
n_fft = 350
frame_size = 350
hop_size = frame_size // 2
freq_mask_scale= 50
time_mask_scale= 80
text_mapping = TextMapping()



def padding(melspecs, labels):
    '''
    Add the paddings to the batch of input data to keep all data in same batch have same sequence length

    Parameters
    ----------
    melspecs: Numpy.ndarray
        The multi-dimensional array that contains the mel-spectrogram as features for every input data
    labels: Numpy.ndarray
        The multi-dimensioanl array that contains ground truth labels

    Return
    ------
    (Numpy.ndarray, Numpy.ndarray)
        The function returns the mel-spectrogram and ground truth label with padded length

    '''

    melspecs = nn.utils.rnn.pad_sequence(melspecs, batch_first= True).unsqueeze(1).transpose(2,3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return melspecs, labels


def get_MelSpectrogram(sample_rate,n_fft, mels_number, frame_size, hop_size, f_min = 0):
    '''
    Steps in Featuren Engineering. The function extracts the mel-spectrogram from audio input with set hyperparameters

    Parameters
    ----------
    sample_rate: int
        The number indicates the sampling rate, or number of samples conatined in one frames, of the audio data
    mels_number: int
        The number indicates the dimensions of mel-spectrogram
    frame_size: int
        The number indicates the size, or length of one frame
    hop_size: int
        The number stands for the length between the start of this frame and the start of next frame
    


    Return
    ------
    Numpy.ndarray
        The function returns the mel-spectrogram extracted from the pipeline

    '''

    melspec_transform = transform.MelSpectrogram(sample_rate= sample_rate, n_mels=mels_number, frame_size = frame_size, hop_size = hop_size)

    return melspec_transform


def melspec_augmentation(freq_mask_scale, time_mask_param = 80):

    '''
    Pipeline of audio augmentation is implemented here. The augmentation applies masking on the spectrogram to 
    prevent overfitting

    Parameters
    ----------
    freq_mask_scale: int
        The number to decide the range, or size of the masking area along frequency axis in mel-spectrogram
   time_mask_param: int
         The number to decide the range, or size of the masking area along time axis in mel-spectrogram

    


    Return
    ------
    Numpy.ndarray
        The function returns the mel-spectrogram that applied frequency and timing masking

    '''

    augmentation_seq = nn.Sequential(
        transform.FrequencyMasking(freq_mask_param = freq_mask_scale),
        transform.TimeMasking(time_mask_param =time_mask_param)
    )

    return augmentation_seq

def data_preprocessing(data_raw):
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

    melspectrograms = []
    labels = []
    melspec_length = []
    label_length = []

    for (waveform, _, label, _, _,_) in data_raw:
        mel_spec_transform = get_MelSpectrogram(sample_rate, n_fft, n_mels,frame_size, hop_size) #extract the Mel-scale spectrogram from waveform
        mel_spec = mel_spec_transform(waveform)
        aug_seq = melspec_augmentation(freq_mask_scale, time_mask_scale)
        mel_spec = aug_seq(mel_spec) #need applying data augmentation process to the mel spec 
        mel_spec = mel_spec.squeeze(0).transpose(0,1) #(channel, n_mels, time) -> time, n_mels

        melspectrograms.append(mel_spec)

        label = text_mapping.convert_TextToInt(label)
        #print(label)
        label = torch.Tensor(label)
        labels.append(label)

        melspec_length.append(mel_spec.shape[0] // 2) 
        label_length.append(len(label))

    melspecs, labels = padding(melspectrograms, labels)

    return melspecs, labels, melspec_length, label_length

 



def valid_preprocessing(valid_data):
    '''
    The function that takes in the raw input data and process the raw data to be suitable for later training.The 
    feature engineering is applied inside the function. The function aims the preprocessing for validation, instead 
    of training. 

    Parameters
    ----------
    valid_data: Numpy.ndarray
        The ndarray stands for the input data that load from the dataset

    Return
    ------
    Numpy.ndarray, Numpy.ndarray, list(int), list(int)
        The function returns the mel-spectrogram after preprocessing pipeline, ground truth label, and lengths of both.

    '''

    melspectrograms = []
    labels = []
    melspec_length = []
    label_length = []

    for (waveform, _, label, _, _,_) in valid_data:

        mel_spec_transform = get_MelSpectrogram(sample_rate, n_fft, n_mels,frame_size, hop_size) #extract the Mel-scale spectrogram from waveform
        mel_spec  = mel_spec_transform(waveform)
        mel_spec = mel_spec.squeeze(0).transpose(0,1) #(channel, n_mels, time) -> time, n_mels

        melspectrograms.append(mel_spec)
        label = text_mapping.convert_TextToInt(label)
        label = torch.Tensor(label)
        labels.append(label)

        melspec_length.append(mel_spec.shape[0] // 2) 
        label_length.append(len(label))

    melspecs, labels = padding(melspectrograms, labels)

    return melspecs, labels, melspec_length, label_length

    


    

        


