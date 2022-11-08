import librosa
import os
import numpy as np
import dataPreprocess
import SignalProcess


#Apply train and validation split here.
sample_path = '/Volumes/SamSung disk/SubsetData/'
train_dest_path = '/Volumes/SamSung disk/trainSubset/'
valid_dest_path = '/Volumes/SamSung disk/validSubset/'

train_csv_path = '/Volumes/SamSung disk/subset_train.csv'
valid_csv_path = '/Volumes/SamSung disk/subset_valid.csv'
whole_data_csv = '/Volumes/SamSung disk/train-all.csv'



#call the train_valid_split function and get the indexes 
train_index, valid_index = dataPreprocess.train_valid_split(sample_path, train_dest_path, valid_dest_path)


print(train_index)
print(valid_index)
print(len(train_index))
print(len(valid_index))

#Apply the extract corresponding csv

dataPreprocess.get_Subset_Label_from_CSV(whole_data_csv, train_csv_path,valid_csv_path, train_index,valid_index)
