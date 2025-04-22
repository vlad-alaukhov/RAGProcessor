import os
import re

root_folder = "/home/home/Projects/Uraltest"

folders = [os.path.join(root_folder, name) for name in os.listdir(root_folder)]

for folder in folders:
    subfolders = [os.path.join(folder, name) for name in os.listdir(folder)]
    for subfolder in subfolders:
        os.rename(subfolder, re.sub(r'\s+', '_', subfolder))