class TextMapping():

    '''
    A class to build the mapping relationship between character, symbols and indexes, and also inplement the functions to 
    convert the indexs and characters in both directions

    Attributes
    ----------
    self.chars: list(str)
        A list of strings that represent all characters in English plus the space and quote symbol
    self.index: list(int)
        A list of indexs ranges from 1 to the length of the list of self.chars

    Methods
    -------
    convert_TextToInt(self, text_sequence)
        The function implements the conversion from the text sequence in string format, to a list of indexes 
    convert_IntToText(self, label_indexes)
        The function implements the conversion from the labels in indexs in int format, to a list of string

    '''
    def __init__(self):

        self.chars = ['\'', ' ',
             'a', 'b', 
             'c', 'd', 
             'e', 'f', 
             'g', 'h', 
             'i', 'j', 
             'k', 'l', 
             'm', 'n', 
             'o', 'p', 
             'q', 'r', 
             's', 't', 
             'u', 'v', 
             'w', 'x', 
             'y', 'z']
        self.index = [i for i in range(len(self.chars))]
        self.char2Int = {}
        self.int2Char = {}
        for idx in range(len(self.index)):
            self.char2Int[self.chars[idx]] = idx
            self.int2Char[idx] = self.chars[idx]
            
        
    def convert_TextToInt(self, text_sequence):

        '''
        The function implements the conversion from  list of string to labels in integer format

        Parameters
        ----------
        text_sequence: str
            The string of a particular text sequence 

        Return 
        ------
        String
            The function returns a list of integer, or indexs representing the input text sequence
        '''
        
        text_sequence = text_sequence.lower()
        sequence_index = []
        for char in text_sequence:
            char_idx = self.char2Int[char]
            sequence_index.append(char_idx)
            
        return sequence_index
    
    def convert_IntToText(self, label_indexes):
        
        '''
        The function implements the conversion from the labels in indexs in int format, to a list of string

        Parameters
        ----------
        label_indexes: list(int)
            The list of integers that representing different characters in a particular sequence

        Return 
        ------
        String
            The function returns a string of sequence converted from indexes
        '''
        
        sequence_text = []
        for index in label_indexes:
            sequence_text.append(self.int2Char[index])
            
        return ''.join(sequence_text)
    