import scipy.io as scio
import os
import numpy as np

vib_folder = 'vib_signals'
filelist = [f for f in os.listdir(vib_folder) if f.endswith('.mat')]
npylist = []
for f in filelist:
    data = scio.loadmat(f'{vib_folder}/{f}')
    print('keys:\n',data.keys())
    key = f.split('.')[0]
    print(f'{key} shape:', data[key].shape)
    npylist.append(data[key])
resized = np.stack([li for li in npylist], axis=1)
print(f'resized data shape: {resized.shape}')
print(resized[3])