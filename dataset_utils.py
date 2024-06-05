import os
import h5py
from tqdm import tqdm

def count_events(dataset_path):
    with h5py.File(dataset_path, "r") as dataset:
        n_samples = dataset['stimulus_identities'][:].shape[0]
    return n_samples

def extract_event_by_idx(dataset_path, idx):
    with h5py.File(dataset_path, "r") as dataset:
        start,end = dataset['length_idx'][idx : idx + 2]
        aud = dataset["audio"][start:end, ...]
        loc = dataset["locations"][idx]
        stim_id = dataset["stimulus_identities"][idx]
    return aud, loc, stim_id

def extract_speaker_names(dataset_path):
    with h5py.File(dataset_path, "r") as dataset:
        stimulus_names = dataset['stimulus_names'][:]
    return stimulus_names

def extract_events(dataset_path, output_path, idx_list):    
    
    if not os.path.exists(output_path):
        with h5py.File(output_path, "w") as f:
            f.create_dataset("audio", shape=(0,4), maxshape=(None,4))
            f.create_dataset("length_idx", shape=(1,), data=[0], maxshape=(None,))
            f.create_dataset("locations", shape=(0,2), maxshape=(None,2))
            f.create_dataset("stimulus_identities", shape=(0,), maxshape=(None,))
            stimulus_names = extract_speaker_names(dataset_path)
            f.create_dataset("stimulus_names", shape=stimulus_names.shape, data=stimulus_names)
        
    with h5py.File(output_path, "a") as f:
        for idx in tqdm(idx_list):
            aud, loc, stim_id = extract_event_by_idx(dataset_path, idx)

            aud_len = aud.shape[0]
            
            audio = f['audio']
            audio.resize(audio.shape[0] + aud_len, axis=0)
            audio[-aud_len:] = aud
            
            length_idx = f['length_idx']
            length_idx.resize(length_idx.shape[0]+1, axis=0)
            length_idx[-1:] = length_idx[-2:-1][0] + aud_len
            
            locations = f['locations']
            locations.resize(locations.shape[0]+1, axis=0)
            locations[-1:] = loc

            stimulus_identities = f['stimulus_identities']
            stimulus_identities.resize(stimulus_identities.shape[0]+1, axis=0)
            stimulus_identities[-1:] = stim_id