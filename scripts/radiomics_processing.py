#Imports
import os
import argparse
import ast

import numpy as np
from radiomics import featureextractor

############################################################################################

#Taking arguments from user
def build_args():
    parser = argparse.ArgumentParser(description='Body Pyradiomics')
    parser.add_argument("--data_path", type=str, default="../../../../../../vol/aimspace/users/wyo/radiomics/body/")
    parser.add_argument("--save_path", type=str, default="../../../../../../vol/aimspace/users/wyo/radiomics/body_values/")

    args = parser.parse_args()
    return args
    
############################################################################################

if __name__ == '__main__':
    args = build_args()

    dirs = next(os.walk(args.data_path))[2]
    
    for dir in dirs:
        path = f'{args.data_path}{str(dir)}'
        data = np.load(path, allow_pickle=True)
        lst = data.files
        
        for item in lst:
            values_list = data[item]

        values_list = values_list.item()

        numerical_values = []

        for item in values_list:
            numerical_values.append(values_list[item])

        numerical_values = np.asarray(numerical_values)

        radiomics = []

        for item in numerical_values:
            try:
                if(type(item) == tuple):
                    for single in item:
                        radiomics.append(float(single))
                else:
                    radiomics.append(float(item))
            except:
                print(item)

        radiomics = np.asarray(radiomics)
        save_path = f'{args.save_path}{str(dir)}'

        np.savez(save_path, radiomics)






