#imports
import os
import argparse
import ast
import pickle as pkl
from itertools import tee
import random

import numpy as np
import pandas as pd
import sys
from scipy.stats import randint
from sklearn.decomposition import PCA

############################################################################################

#Taking arguments from user
def build_args():
    parser = argparse.ArgumentParser(description='GAE')
    parser.add_argument("--organ_latent_spaces_path", type=str, default="../../../../../../vol/aimspace/users/wyo/latent_spaces/vertices_prediction/")
    parser.add_argument("--body_latent_spaces_path", type=str, default="../../../../../../vol/aimspace/users/wyo/latent_spaces/body_vertices_prediction/")
    parser.add_argument("--pca", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="../../../../../../vol/aimspace/users/wyo/latent_spaces/combined/")
    parser.add_argument("--organ_radiomics_path", type=str, default="../../../../../vol/aimspace/projects/ukbb/whole_body/organ_segmentations/features/radiomics_features_all_enhanced/")
    parser.add_argument("--body_radiomics_path", type=str, default="../../../../../../vol/aimspace/users/wyo/radiomics/body_values/")
    parser.add_argument("--body", type=ast.literal_eval, default=False)
    parser.add_argument("--organ_radiomics", type=ast.literal_eval, default=False)
    parser.add_argument("--body_radiomics", type=ast.literal_eval, default=False)
    
    args = parser.parse_args()
    return args

def get_size(path, unit):
    exponents_map = {'bytes': 0, 'kb': 1, 'mb': 2, 'gb': 3}
    
    file_size = os.path.getsize(path)
    size = file_size / 1024 ** exponents_map[unit]
    return size
    
############################################################################################

if __name__ == '__main__':
    args = build_args()
    organs = ["liver", "spleen", "left_kidney", "right_kidney", "pancreas"]
    radiomics_organs = {"liver": "liv.npz", "pancreas": "pnc.npz", "spleen": "spl.npz", "left_kidney": "lkd.npz", "right_kidney" : "rkd.npz"}
    dirs = next(os.walk(f'{args.organ_latent_spaces_path}{organs[0]}'))[2]
    errors = []

    if(args.pca > 0):
        if(args.body):
            save_path = f'{args.save_path}organs_and_body_{args.pca}_pca'
        else:
            save_path = f'{args.save_path}organs_{args.pca}_pca'
    else:
        if(args.body):
            save_path = f'{args.save_path}organs_and_body'
        else:
            save_path = f'{args.save_path}organs'

    if(args.pca > 0):
        pca = PCA(n_components=args.pca)

    if(not args.body):
        for dir in dirs:
            try:
                initial = True
                if(not os.path.exists(f'{save_path}/{dir}')):
                    for organ in organs:
                        organ_path = f'{args.organ_latent_spaces_path}{organ}/{str(dir)}'
                        with open(organ_path, "rb") as fp:
                            vertices_latent_space = pkl.load(fp)
                        fp.close()
                        if(initial):
                            initial = False
                            if(args.pca > 0):
                                vertices_latent_space = pca.fit_transform(vertices_latent_space)
                                vertices_latent_space = vertices_latent_space

                            combined_latent_space = vertices_latent_space
                        else: 
                            if(args.pca > 0):
                                vertices_latent_space = pca.fit_transform(vertices_latent_space)
                                vertices_latent_space = vertices_latent_space

                            combined_latent_space = np.concatenate((combined_latent_space, vertices_latent_space))

                    #WRITING
                    with open(f'{save_path}/{dir}', "wb") as fp:
                        pkl.dump(combined_latent_space, fp)
                    fp.close()

                    file_size = get_size(f'{save_path}/{dir}', 'mb')
                    if(file_size > 6):
                        print(f'dir: {dir}, size: {file_size}, data shape: {combined_latent_space.shape}, data: {combined_latent_space}')
                        sys.exit()
            except:
                errors.append(dir)

    if(args.body):
        dirs = next(os.walk(args.body_latent_spaces_path))[2]

        for dir in dirs:
            try:
                body_path = f'{args.body_latent_spaces_path}{str(dir)}'
                if(args.pca > 0):
                    combined_organs_path = f'../../../../../../vol/aimspace/users/wyo/latent_spaces/combined/organs_{args.pca}_pca/{str(dir[:-4])}' #PCA Number sae as organ!!!
                else:
                    combined_organs_path = f'../../../../../../vol/aimspace/users/wyo/latent_spaces/combined/organs/{str(dir[:-4])}'

                with open(body_path, "rb") as fp:
                    body_vertices_latent_space = pkl.load(fp)
                fp.close()

                with open(combined_organs_path, "rb") as fp:
                    organ_vertices_latent_space = pkl.load(fp)
                fp.close()

                if(args.pca > 0):
                    body_vertices_latent_space = pca.fit_transform(body_vertices_latent_space)
                    body_vertices_latent_space = body_vertices_latent_space

                combined_latent_space = np.concatenate((body_vertices_latent_space, organ_vertices_latent_space))

                #WRITING
                with open(f'{save_path}/{dir[:-4]}', "wb") as fp:
                    pkl.dump(combined_latent_space, fp)
                fp.close()

                file_size = get_size(f'{save_path}/{dir[:-4]}', 'mb')
                if(file_size > 7):
                    print(f'dir: {dir}, size: {file_size}, data shape: {combined_latent_space.shape}, data: {combined_latent_space}')
                    sys.exit()
            except:
                errors.append(dir)
                
    if(args.organ_radiomics or args.body_radiomics):
        
        if(args.organ_radiomics and args.body_radiomics):
            dirs = next(os.walk('../../../../../../vol/aimspace/users/wyo/latent_spaces/combined/organs_and_body/'))[2]
            all_and_all_save_path = f'{args.save_path}all_and_all_radiomics'
            all_radiomics_save_path = f'{args.save_path}all_radiomics'

            for dir in dirs:
                try:
                    body_radiomics_path = f'{args.body_radiomics_path}{str(dir)}.npz'
                    organ_radiomics_path = f'{args.organ_radiomics_path}{str(dir)}' #folder not files
                    combined_all_path = f'../../../../../../vol/aimspace/users/wyo/latent_spaces/combined/organs_and_body/{str(dir)}'

                    #combine organs radiomics
                    combined_organ_radiomics = np.asarray([])
                    first = True
                    for organ in organs:
                        data_path = os.path.join(organ_radiomics_path, radiomics_organs[organ])

                        if(first):
                            first = False
                            data = np.load(data_path)
                            lst = data.files
                            for item in lst:
                                combined_organ_radiomics = data[item]
                        else:
                            data = np.load(data_path)
                            lst = data.files
                            for item in lst:
                                combined_organ_radiomics = np.append(combined_organ_radiomics, data[item])

                    #body radiomics
                    body_radiomics = np.asarray([])
                    data_path = body_radiomics_path
                    data = np.load(data_path)
                    lst = data.files
                    for item in lst:
                        body_radiomics = data[item]

                    all_radiomics = np.append(body_radiomics, combined_organ_radiomics)
                    
                    with open(combined_all_path, "rb") as fp:
                        combined_all = pkl.load(fp)
                    fp.close()
                    all = np.append(all_radiomics, combined_all)

                    #WRITING
                    with open(f'{all_radiomics_save_path}/{dir}', "wb") as fp:
                        pkl.dump(all_radiomics, fp)
                    fp.close()

                    #WRITING
                    with open(f'{all_and_all_save_path}/{dir}', "wb") as fp:
                        pkl.dump(all, fp)
                    fp.close()

                except:
                    errors.append(dir)

        elif(args.organ_radiomics):
            dirs = next(os.walk('../../../../../../vol/aimspace/users/wyo/latent_spaces/combined/organs_and_body/'))[2]
            all_and_organ_radiomics_save_path = f'{args.save_path}all_and_organ_radiomics'

            for dir in dirs:
                try:
                    organ_radiomics_path = f'{args.organ_radiomics_path}{str(dir)}' #folder not files
                    combined_all_path = f'../../../../../../vol/aimspace/users/wyo/latent_spaces/combined/organs_and_body/{str(dir)}'

                    #combine organs radiomics
                    combined_organ_radiomics = np.asarray([])
                    first = True
                    for organ in organs:
                        data_path = os.path.join(organ_radiomics_path, radiomics_organs[organ])

                        if(first):
                            first = False
                            data = np.load(data_path)
                            lst = data.files
                            for item in lst:
                                combined_organ_radiomics = data[item]
                        else:
                            data = np.load(data_path)
                            lst = data.files
                            for item in lst:
                                combined_organ_radiomics = np.append(combined_organ_radiomics, data[item])

                    with open(combined_all_path, "rb") as fp:
                        combined_all = pkl.load(fp)
                    fp.close()
                    all_and_organ_radiomics = np.append(combined_organ_radiomics, combined_all)

                    #WRITING
                    with open(f'{all_and_organ_radiomics_save_path}/{dir}', "wb") as fp:
                        pkl.dump(all_and_organ_radiomics, fp)
                    fp.close()

                except:
                    errors.append(dir)


        elif(args.body_radiomics):
            dirs = next(os.walk('../../../../../../vol/aimspace/users/wyo/latent_spaces/combined/organs_and_body/'))[2]
            all_and_body_radiomics_save_path = f'{args.save_path}all_and_body_radiomics'

            for dir in dirs:
                try:
                    body_radiomics_path = f'{args.body_radiomics_path}{str(dir)}.npz'
                    combined_all_path = f'../../../../../../vol/aimspace/users/wyo/latent_spaces/combined/organs_and_body/{str(dir)}'

                    #body radiomics
                    body_radiomics = np.asarray([])
                    data_path = body_radiomics_path
                    data = np.load(data_path)
                    lst = data.files
                    for item in lst:
                        body_radiomics = data[item]

                    with open(combined_all_path, "rb") as fp:
                        combined_all = pkl.load(fp)
                    fp.close()
                    all_and_body_radiomics = np.append(body_radiomics, combined_all)

                    #WRITING
                    with open(f'{all_and_body_radiomics_save_path}/{dir}', "wb") as fp:
                        pkl.dump(all_and_body_radiomics, fp)
                    fp.close()

                except:
                    errors.append(dir)

                

    print(errors)