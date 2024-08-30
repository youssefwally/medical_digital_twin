#imports
import os
import argparse
import numpy as np
import pandas as pd

############################################################################################

#Taking arguments from user
def build_args():
    parser = argparse.ArgumentParser(description='GAE')
    parser.add_argument("--meshes_path", type=str, default="../../../../../../vol/space/projects/ukbb/projects/silhouette/gender_registered_1k/")
    parser.add_argument("--save_path", type=str, default="../data/")
    parser.add_argument("--split_ratio", nargs='+', type = int, default=[0.8, 0.1, 0.1], help="[train, val, test] as decimals")
    
    args = parser.parse_args()
    return args
    
############################################################################################

def split_data(meshes_path, save_path, split_ratio):

    dirs = next(os.walk(meshes_path))[2]
    dirs_np = np.asarray(dirs)

    np.random.shuffle(dirs_np)

    train_arr, test_arr = np.split(dirs_np, [int((split_ratio[0]+split_ratio[1])*len(dirs_np))])
    train_arr, val_arr = np.split(train_arr, [int((1-split_ratio[1])*len(train_arr))])
    print(type(np.asarray(train_arr)))

    np.savetxt(f'{save_path}body_split_train.txt', train_arr, fmt='%s')
    np.savetxt(f'{save_path}body_split_val.txt', val_arr, fmt='%s')
    np.savetxt(f'{save_path}body_split_test.txt', test_arr, fmt='%s')

    return [len(train_arr), len(val_arr), len(test_arr)]
    
############################################################################################

if __name__ == '__main__':
    args = build_args()

    splits_sizes = split_data(args.meshes_path, args.save_path, args.split_ratio)

    print(splits_sizes)