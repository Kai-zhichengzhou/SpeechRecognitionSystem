
import torch
from ..mapping import TextMapping as mapping


#define WER and CER for evalution in ASR

text_mapping = mapping.TextMapping()


def min_edit_distance(pred,label):

    '''
    Calculate the minimal edit distance (word level) between two sequences

    Parameters
    ----------
    pred: list(str)
        A list of predicted words that contains in a sequence

    label: list(str)
        A list of ground-truch label words that contains in a sequence

    Return
    ------
    int
        returns the smallest word edit distance between two sequence 

    '''
    if pred == label:
        return 0

    if len(pred) == 0:
        return len(label)

    if len(label) == 0:
        return len(pred)

    h = len(pred) + 1
    w = len(label) + 1

    dp = [[ 0 for i in range(w)] for j in range(h)]

    for i in range(h):
        
        dp[i][0] = h

    for j in range(w):
        dp[0][j] = j

    for i in range(1, h):

        for j in range(1, w):

            if pred[i - 1] == label[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            else:
                dp[i][j] = min(dp[i -1][j - 1], dp[i][j - 1], dp[i - 1][j]) + 1

    return dp[h - 1][w - 1]

    




def calc_word_error_rate(label_seq, pred_seq):

    '''
    Calculate the word error rate (WER) to evaluate the performance of prediction

    Parameters
    ----------
    pred_seq: str
        A string of predicted words as a sequence

    label: lstr
        A string of ground-truch words as a sequence

    Return
    ------
    float
        returns WER of the two input sequence
    '''

    if len(label_seq) == 0:
        return "Error of length of label. Label should not be empty"

    pred_seq = pred_seq.lower()
    label_seq = label_seq.lower()

    pred_seq = pred_seq.split(' ')
    label_seq = label_seq.split(' ')

    edit_distance = min_edit_distance(pred_seq, label_seq)
    print(edit_distance)
    print(pred_seq)
    print(label_seq)

    WER = float(edit_distance) / len(label_seq)
    print(WER)
    return WER


def calc_char_error_rate(label_sequence, pred_sequence):

    '''
    Calculate the character error rate (CER) to evaluate the performance of prediction

    Parameters
    ----------
    pred_seq: str
        A string of predicted words as a sequence

    label: lstr
        A string of ground-truch words as a sequence

    Return
    ------
    float 
        returns CER of the two input sequence
    '''

    if len(label_sequence) == 0:
        return "Error of the length of Label Sequence. Label should not be empty."
    pred_sequence = pred_sequence.lower()
    label_sequence = label_sequence.lower()

    #remove the space is necessary

    pred_sequence = ''.join(pred_sequence.split(' '))
    label_sequence = ''.join(label_sequence.split(' '))
    distance = min_edit_distance(pred_sequence,label_sequence)

    CER = float(distance) / len(label_sequence)

    return CER



    

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

