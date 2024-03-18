# %% Include epoching according to events. 
import os

import torch
import numpy as np
import pandas as pd
import autoreject

import mne
from mne_bids import BIDSPath

import scipy.io
import scipy
import scipy.signal

from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
plt.style.use('dark_background')

import re
import multiprocessing
from joblib import Parallel, delayed


from collections import Counter

import random

# mae3, mit source estimation. 

# %%
parallelize = False # true sometimes runs into errors? 
n_jobs = 16

data_path = "/vol/aimspace/users/dena/Documents/clean_brain_age/brain-age-benchmark/bids_LEMON/data"

save_path = "/vol/aimspace/users/dena/Documents/mae/data/lemon_epoched_by_events"

epoch_division_style =  "consider_events_10_seconds" # "seperate_data_in_10_sec_epochs"

SMALL_DATASET = True
dataset_size = 20 
if SMALL_DATASET: 
    save_path +=f"_small_{dataset_size}"

if not os.path.exists(save_path):  
    os.makedirs(save_path)




# %%
participant_tab_data = pd.read_csv(os.path.join(data_path, "participants.csv"), sep="\t", index_col=0) 
participant_tab_data['sex'] = ['M' if participant_tab_data.at[index, 'Gender_ 1=female_2=male'] == 2 else 'F' for index in participant_tab_data.index[:]]


# %%
labels_dict = {}
for idx in range(len(participant_tab_data)):
    key = participant_tab_data.index[idx]
    sex = participant_tab_data.iloc[idx]["sex"]
    age = participant_tab_data.iloc[idx]["age"] #.split("-")], dtype=np.float32).mean()

    labels_dict[key] = (age, sex)
print(len(labels_dict))

torch.save(labels_dict, os.path.join(save_path, "labels_dict.pt"))

# %% [markdown]
# ### Create the raw dataset

# %%
def get_all_files(rootdir): #, substring):
    files = []
    for file in os.listdir(rootdir):
        curr_object = os.path.join(rootdir, file)

        if os.path.isdir(curr_object) and (curr_object.split("-")[0] == "sub" or curr_object == "eeg"):
            files += get_all_files(curr_object) #, substring)
        # elif substring in file and ".set" in file:
        #     files.append(curr_object)

    return files

# %%
first_iteration_subj_ids = []
files = []
edf_files = []
channels_tsv = []

for file in os.listdir(data_path):
    # print(file)
    first_iteration_subj_ids.append(file)

for subj in first_iteration_subj_ids: 
    curr_path = os.path.join(os.path.join(data_path, subj, "eeg"))
    if os.path.isdir(curr_path): 
        buffer = 0
        for file in os.listdir(curr_path):
            if file.split(".")[-1] == "vhdr": 
                edf_files.append(os.path.join(curr_path, file))
                buffer += 1
        if buffer == 0: 
            print("EEG folder but no vhdr file?: " + file)
            
            
        if file.split("_")[-1] == "channels.tsv": 
            channels_tsv.append(file)

print(len(edf_files))

# %%
if SMALL_DATASET: 
    edf_files = edf_files[:dataset_size]

print(len(edf_files))

# %%
analyze_channels = [
  'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
  'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'AFz',
  'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'AF7',
  'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT7', 'FC3', 'FC4', 'FT8',
  'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2',
  'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8']


# %%
fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)
subject = 'fsaverage'
trans = 'fsaverage'  # fsaverage has a built-in identity transform

# %%
montage = mne.channels.make_standard_montage('standard_1005') # for LEMON
# all three are independent of the EEG. Since we are using fsaverage, there is no need to seperate between individuals (i.e. to include it in loop) 
src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
model = mne.make_bem_model(subject=subject, subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

# %%
epoch_length = 10  # in seconds
SAVE = True
path_to_save = save_path

# %%
l_freq = 0.1 #lemon and tuab
h_freq = 49 #lemon and tuab

resample_freq = 200 # Hz for lemon and tuab
eog_channels = ["Fp1"] # only for lemon


# %% [markdown]
# 
# ValueError: DigMontage is only a subset of info. There is 1 channel position not present in the DigMontage. The channel missing from the montage is:
# 
# ['VEOG'].
# 
# Consider using inst.rename_channels to match the montage nomenclature, or inst.set_channel_types if this is not an EEG channel, or use the on_missing parameter if the channel position is allowed to be unknown in your analyses.

skipped_subjects = []


def process_edf_file(edf_filename):
# for edf_filename in edf_files: 
    subject_id = edf_filename.split("/")[-1].split("_")[0].split("-")[1]

    if not os.path.exists(f'{path_to_save}/{subject_id}/epochs-epo.fif'):  
        print(f"Processing {subject_id}. ")
        
        # Read the EEG data from the .edf file
        raw = mne.io.read_raw_brainvision(edf_filename, verbose=False, preload=True)
        
        # VEOG is not in montage. Config file of LEMON does something with Fp1 - however, this channel already exists...
        # What to do? 
        # mne.rename_channels(raw.info, {'Fp1': 'VEOG'})
        raw.drop_channels(["VEOG"])

        raw.set_montage(montage)

        # Check if all channels are there
        to_analyze_channels_not_in_raw_ch_names = list(set(analyze_channels) - set(raw.ch_names))
        if to_analyze_channels_not_in_raw_ch_names:
            raise ValueError(f"Subject {subject_id} does not have all necessary channels. Missing channels: {to_analyze_channels_not_in_raw_ch_names}")

        raw.pick(analyze_channels, verbose=False).resample(sfreq=resample_freq, verbose=False)
        raw.filter(l_freq, h_freq)

        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0)
        # EEG average reference (using a projector) is mandatory for modeling (inverse etc) >> will otherwise raise error later on. 
        # Config file basically took every channel as reference, as far as I could gather
        raw.set_eeg_reference('average', projection=True)
        raw.apply_proj()

        if epoch_division_style == "seperate_data_in_10_sec_epochs":
            ### Divide recording into 10 second epochs:
            # Create events array: [[sample1, 0, event_id], [sample2, 0, event_id], ...]
            event_id = 1  # could be any integer as event id
            num_samples = len(raw.times)
            events = np.array([
                [i, 0, event_id] for i in range(0, num_samples, int(raw.info['sfreq'] * epoch_length))
            ])
            epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=epoch_length, baseline=None, preload=True)

        elif epoch_division_style == "consider_events_10_seconds": 
            events, event_id = mne.events_from_annotations(raw)
            filtered_event_ids = {key: value for key, value in event_id.items() if value in events[:, -1]}
            # epochs = mne.Epochs(raw, events, event_id={'New Segment/': 99999, 'Stimulus/S  1': 1, 'Stimulus/S200': 200, 'Stimulus/S210': 210}, tmin=-0.2, tmax=10, baseline=(None, 0), preload=True)
            epochs = mne.Epochs(raw, events, event_id=filtered_event_ids, tmin=-0.2, tmax=10, baseline=(None, 0), preload=True, event_repeated='merge')
              
        else: 
            print("ERROR!! epoch_division_style is not available.")
        
        
        #########################
        #  EXPERIMENT (NOT IN MNE BIDS PIPELINE): NORMALIZE EPOCHS!

        # Mean, std, z-score normalization
        data = epochs.get_data()  # 3D array with (n_epochs, n_channels, n_times)
        mean = np.mean(data, axis=(0, 2), keepdims=True)
        std = np.std(data, axis=(0, 2), keepdims=True)
        normalized_data = (data - mean) / std

        # Replace data in epochs
        epochs._data = normalized_data

        #########################

        cov = mne.make_ad_hoc_cov(raw.info) 

        # Those two lines are from the Benchmarking algo
        ar = autoreject.AutoReject(n_jobs=1, cv=5)
        epochs = ar.fit_transform(epochs)

        inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, loose=0.2, depth=0.8)
        #config file says this is unnecessary? 
        stc = mne.minimum_norm.apply_inverse(epochs.average(), inv, lambda2=1.0 / 9.0, method='dSPM')

        if SAVE:
            if not os.path.exists(os.path.join(path_to_save, subject_id)):
                os.makedirs(os.path.join(path_to_save, subject_id))
            epochs.save(f'{path_to_save}/{subject_id}/epochs-epo.fif', overwrite=True)
            stc.save(f'{path_to_save}/{subject_id}/source_estimate', overwrite=True)
            mne.write_forward_solution(f'{path_to_save}/{subject_id}/forward_model-fwd.fif', fwd, overwrite=True)
            mne.minimum_norm.write_inverse_operator(f'{path_to_save}/{subject_id}/inverse_operator-inv.fif', inv, overwrite=True)
    else: 
        skipped_subjects.append(subject_id)
        print(f"Skipped {subject_id}. ")

    

if __name__ == "__main__": #
    if parallelize: 
        pool = multiprocessing.Pool(processes=n_jobs)

        # Parallel processing using multiprocessing.Pool
        pool.map(process_edf_file, edf_files)
        # Close the pool of processes
        pool.close()
        pool.join()
    
    else: 
        for edf_filename in edf_files: 
            process_edf_file(edf_filename)



# %%
print("finished something")


# %%
