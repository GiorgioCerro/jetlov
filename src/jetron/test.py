from lundnet.dgl_dataset import DGLGraphDatasetLund as Dataset
from pathlib import Path

path = "/scratch/gc2c20/data/test/"
dataset = Dataset(filepath_bkg=Path(path+"test_QCD_500GeV.json.gz"),
                filepath_sig=Path(path+"test_WW_500GeV.json.gz"),
                nev=-1, n_samples=100)

print(dataset[0])
