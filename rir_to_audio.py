import numpy as np
import h5py
import os
from tqdm import tqdm
import soundfile as sf
import numpy as np
import time
from pqdm.processes import pqdm as pqdm_p
from pqdm.threads import pqdm as pqdm_t

from torch import from_numpy
from torchaudio import functional as AF

from scipy.signal import convolve

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "save_path",
    type=str,
    help="path to save audio dataset",
)

parser.add_argument(
    "--rir-path",
    dest="rir_path",
    type=str,
    help="path to load rir dataset",
    required=False,
    default="/vast/ci411/gerbil_data/rir_datasets/default.h5",
)

parser.add_argument(
    "--voc-path",
    dest="voc_path",
    type=str,
    help="path to load vocalizations",
    required=False,
    default="/vast/ci411/gerbil_data/vocal_datasets/raw/speaker_resampled",
)

parser.add_argument(
    "--n-jobs",
    dest="n_jobs",
    type=int,
    help="number of jobs to use in parallized process",
    required=False,
    default=16,
)

parser.add_argument(
    "--all-vocs",
    dest="all_vocs",
    type=bool,
    help="use all vocalizations per rir (if not, choose one random per rir)",
    required=False,
    default=False,
)
    
parser.add_argument(
    "--limit",
    dest="limit",
    type=int,
    help="limit of RIRs to use",
    required=False,
    default=None,
)

parser.add_argument(
    "--use-threads",
    dest="use_threads",
    type=bool,
    help="use threads for parallelization (otherwise, use processes)",
    required=False,
    default=True,
)
    

args = parser.parse_args()


def store_result(results, save_path, stimulus_names=None):
    
    with h5py.File(save_path, "w") as f:
        audio = f.create_dataset("audio", shape=(0,4), maxshape=(None,4))
        length_idx = f.create_dataset("length_idx", shape=(1,), data=np.array([0]), maxshape=(None,))
        locations = f.create_dataset("locations", shape=(0,2), maxshape=(None,2))
        stimulus_identities = f.create_dataset("stimulus_identities", shape=(0,), maxshape=(None,))
        if stimulus_names is not None:
            f.create_dataset("stimulus_names", shape=stimulus_names.shape, data=stimulus_names)
            
        for i, item in tqdm(enumerate(result)):
            aud_len = item[0].shape[1]
            audio.resize(audio.shape[0] + aud_len, axis=0)
            audio[-aud_len:] = item[0].T
            
            length_idx.resize(length_idx.shape[0]+1, axis=0)
            length_idx[-1:] = length_idx[-2:-1][0] + aud_len
            
            locations.resize(locations.shape[0]+1, axis=0)
            locations[-1:] = item[1][None,:]
            
            stimulus_identities.resize(stimulus_identities.shape[0]+1, axis=0)
            stimulus_identities[-1:] = item[2]

if args.use_threads:
    pqdm = pqdm_t
else:
    pqdm = pqdm_p

#load RIR dataset
rir_dataset = h5py.File(args.rir_path, "r")

#extract locations, rir data, rir indexes
locations = rir_dataset['locations'][:]
rir_all = rir_dataset['rir'][:]
rir_length_idx = rir_dataset['rir_length_idx'][:]
    
rir_dataset.close()


#load vocalizations (normalized)
vocalizations = []
for file in os.listdir(args.voc_path):
    aud, sr = sf.read(os.path.join(args.voc_path, file))
    aud = (aud - np.average(aud)) / np.std(aud)
    vocalizations.append(aud)

#define helper function for extracting RIR/location
def rir_loc_for_index(idx):
    start, end = rir_length_idx[idx : idx + 2]
    rir = rir_all[start:end, :]
    loc = locations[idx]
    return rir.T, loc

#define function for convolving rir w/ vocalization
def convolve_audio(idx_pair):
    rir_idx, voc_idx = idx_pair
    rir, loc = rir_loc_for_index(rir_idx)
    voc = vocalizations[voc_idx][None, :]
    #audio = AF.convolve(from_numpy(rir), from_numpy(voc), mode="valid").numpy()
    audio = convolve(rir, voc, mode="full")
    return audio, loc, voc_idx


#either convolve with each vocalization or one per rir
if args.all_vocs:
    pairs = [(i,j) for i in range(len(rir_length_idx)-1)
                   for j in range(len(vocalizations))]
else:
    pairs = [(i,np.random.randint(0, len(vocalizations))) for i in range(len(rir_length_idx)-1)]

#debugging limit    
if args.limit is not None:
    pairs = pairs[:args.limit]

#convolve rirs/vocalizatoins
t0 = time.time()
result = pqdm(pairs, convolve_audio, n_jobs=args.n_jobs)
t1 = time.time()
print(f"Took {t1-t0:.2f} seconds")

#gather results to appropriate format for .h5 storage
print("Gathering results...")
stimulus_names = np.array([item.split('.')[0] for item in os.listdir(args.voc_path)], dtype=object)

store_result(result, args.save_path, stimulus_names=stimulus_names)