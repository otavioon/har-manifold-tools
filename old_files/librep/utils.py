import os
import requests
import zipfile
import pandas as pd
from tqdm import tqdm

def download_url(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    return fname
            
            
def unzip_file(filename: str, destination: str):
    with zipfile.ZipFile(filename, 'r') as zf:
        for member in tqdm(zf.infolist(), desc='Extracting '):
            zf.extract(member, destination)
            

def load_training_files(root_path: str, 
                        train_fname: str = "train.csv", 
                        val_fname = "val.csv", 
                        test_fname = "test.csv", 
                        loader=pd.read_csv) -> tuple:
    res = []
    for f in [train_fname, val_fname, test_fname]:
        if f is None:
            res.append(None)
        else:
            path = os.path.join(root_path, f)
            values = loader(path)
            res.append(values)
    return tuple(res)
    