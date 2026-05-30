#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:56:53 2024

@author: edward
"""
import os
import io
import json
import tarfile
import webdataset as wds

#%%
df = {
       "row": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
       "A": list("acbdgsftvagadmzk"),
       "B": [0.6, 0.2, 0.3, 0.7, 0.1, 0.3, 0.5, 0.2, 0.6, 0.3, 0.5, 0.9, 0.1, 0.4, 0.5, 0.6],
       "C": [True, False, False, True, True, True, False, False, True, False, True, True, True, False, True, False],
       "D": [[1, 3], [2], [6, 2, 3], [], [], [], [2, 5], [1], [], [3, 5], [3, 4], [4, 5], [4], [], [1, 9], [8]],   
}
df = [dict(zip(df,t)) for t in zip(*df.values())]

# %% Create the archived tar files
# Each example is stored as a json dict
os.makedirs("./tar_dataset", exist_ok=True)
tar = None

for ii, row in enumerate(df):
    if ii % 4 == 0:
        idx = ii // 4
        if tar is not None:
            tar.close()
        tar = tarfile.open(f"./tar_dataset/data-{idx:05d}.tgz", "w:gz")
    data = json.dumps(row).encode("utf-8")
    file_like_object = io.BytesIO(data)
    
    # Write to file
    tarinfo = tarfile.TarInfo(name=f'json-{ii:05d}.json')
    tarinfo.size = len(data)

    # Add the file to the tar archive
    tar.addfile(tarinfo, fileobj=file_like_object)
tar.close()
    
# %% Read the tar file as webdataset
tabular_dataset = (
    wds.WebDataset("./tar_dataset/")
    .to_tuple("json").batched(2)
)
print("Trying to print dataset")
for s in tabular_dataset:
    print(s)
    break

#%%

tabular_dataset = (
    wds.WebDataset("./archive/")
    .to_tuple("json").batched(2)
)
for s in tabular_dataset:
    print(s)
    break




