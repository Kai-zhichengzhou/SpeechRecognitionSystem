import os
import numpy as np

import shutil
import librosa
import pandas as pd


def train_valid_split(data_path, train_dest_path, valid_dest_path, train_set_ratio = 0.7):
    
    data_dir = os.walk(data_path)
  
    train_index_lst = []
    valid_index_lst = []
    for path, dir_name, file_list in data_dir:
    
    #need to get total files in subset to calculate the number for train and valid
    total_data = len(file_list)
    num_train = round(total_data * 0.7)
    num_valid = total_data - num_train
    idx = 0

    for voice_sample in file_list:
        index_str = voice_sample.split('.')[0][-6:]
        index = int(index_str)


        sample_path = os.path.join(path, voice_sample)
      #check the split boundary for train and validation dataset while copying and moving the data
        dest_path = os.path.join(train_dest_path, voice_sample) if idx <= num_train else os.path.join(valid_dest_path, voice_sample)

        train_index_lst.append(index) if idx <= num_train else valid_index_lst.append(index)
        shutil.copy(sample_path, dest_path)

        idx += 1

    return train_index_lst, valid_index_lst


def get_Subset_Label_from_CSV(csv_file, train_csv_dest, valid_csv_dest, train_lst, valid_lst):

  #the data in trainSubset and validSubset are not consecutive
  #necessary to use index. from list above to match the labels in csv
  #extract corresponding labels to form a new subset csv for train and valid both

    whole_df = pd.read_csv(csv_file)

    train_data = {'sample-index': [], 'text': []}
    valid_data = {'sample-index': [], 'text': []}

    df_train = pd.DataFrame(train_data)
    df_valid = pd.DataFrame(valid_data)
  
    for file_idx in train_lst:
        file_name = whole_df.iloc[file_idx].filename.split('.')[0][-13:]
        text = whole_df.iloc[file_idx].text

        new_row = {'sample-index': file_name, 'text':text}

        df_train = df_train.append(new_row, ignore_index = True)

    for file_idx in valid_lst:

        file_name = whole_df.iloc[file_idx].filename.split('.')[0][-13:]
        text = whole_df.iloc[file_idx].text

        new_row = {'sample-index': file_name, 'text':text}

        df_valid = df_valid.append(new_row, ignore_index = True)

    df_train.to_csv(train_csv_dest, encoding='utf-8', index = False)
    df_valid.to_csv(valid_csv_dest,encoding='utf-8', index = False)
