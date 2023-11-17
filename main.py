import CNN_1D as cnn
from PyEMD import CEEMDAN, Visualisation

path = 'vib_signals'
dynamic = cnn.load_files(path).transpose(0, 2, 1)
datas = dynamic[0]
fs = 10000
ceemdan = CEEMDAN()
ceemdan.noise_seed(22)
for S in datas:
    t = len(S)/fs
    ceemdan.ceemdan(S)
    imfs, res = ceemdan.get_imfs_and_residue()
    vis = Visualisation()
    vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
    vis.plot_instant_freq(t, imfs=imfs)
    vis.show()
