import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import pandas as pd
import cv2
from scipy.optimize import curve_fit
import scienceplots
import os
import matplotlib.pyplot as plt

def folder_list(directory_path):
    files_list = []
    for item in os.listdir(directory_path):
        full_path = os.path.join(directory_path, item)
        if os.path.isdir(full_path):
            # print(item)
            files_list.append(item)
    # files = sorted(files_list, key=last_2digits)
    
    return files_list

def last_2digits(x):
    return(int(x.split('_')[-1]))

def last_9chars(x):
    return(x[-7:])

def first_2chars(x):
    return(x[:2])

def conc_data_files(x):
    return(x[-18:])

root = os.getcwd()
plt.style.use(['science', 'nature', 'std-colors'])

base_filepath = '/Volumes/krauss/Sam/GMR4/2024/December/101224/'
conc_files_list = folder_list(base_filepath)
conc_files_list = sorted(conc_files_list, key=first_2chars)

imagePaths = [f for f in glob.glob(f'{base_filepath}{conc_files_list[0]}/Di_water__volts_1_4/Pos0/'+'*.tif')]

img = cv2.imread(imagePaths[0], cv2.IMREAD_UNCHANGED)
r = cv2.selectROI("select the area", img)
image = img[int(r[1]):int(r[1]+r[3]), 
            int(r[0]):int(r[0]+r[2])]

for index, filepath_conc in enumerate(conc_files_list):
    
    filepath = f'{base_filepath}{filepath_conc}/'
    files_list = folder_list(filepath)
    files_list = sorted(files_list, key=last_2digits)
    averages_array = []

    for i in files_list:
        imagePaths = [f for f in glob.glob(f'{filepath}{i}/Pos0/'+'*.tif')]
        files = sorted(imagePaths, key=last_9chars)
        ones_array = np.ones((image.shape[0],image.shape[1],len(files)))

        for i, f in enumerate(files):
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            image = img[int(r[1]):int(r[1]+r[3]), 
                                int(r[0]):int(r[0]+r[2])]

            ones_array[:,:,i] = image

        avg = np.mean(ones_array)
        print(avg)
        averages_array.append(avg)
        
    df = pd.DataFrame(averages_array)
    df.to_csv(f'conc_data/{index}_array_to_csv.csv', index=False)
    print(f'final val = {averages_array}')
