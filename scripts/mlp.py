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

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, cross_val_score, train_test_split
from sklearn.decomposition import PCA

############################################################################################

#Taking arguments from user
def build_args():
    parser = argparse.ArgumentParser(description='GAE')
    parser.add_argument("--edge_path", type=str, default="../../../../../../vol/aimspace/users/wyo/latent_spaces/edge_prediction/")
    parser.add_argument("--vertices_path", type=str, default="../../../../../../vol/aimspace/users/wyo/latent_spaces/vertices_prediction/")
    parser.add_argument("--radiomics_path", type=str, default="../../../../../../vol/aimspace/projects/ukbb/whole_body/organ_segmentations/features/radiomics_features_all_enhanced/")
    parser.add_argument("--body_radiomics_path", type=str, default="../../../../../../vol/aimspace/users/wyo/radiomics/body_values/")
    parser.add_argument("--all_and_organ_radiomics_path", type=str, default="../../../../../../vol/aimspace/users/wyo/latent_spaces/combined/all_and_organ_radiomics/")
    parser.add_argument("--organ", type=str, default="liver")
    parser.add_argument("--label", type=str, default="sex")
    parser.add_argument("--body", type=ast.literal_eval, default=False)
    parser.add_argument("--combine_organs", type=ast.literal_eval, default=False)
    parser.add_argument("--combine_all", type=ast.literal_eval, default=False)
    parser.add_argument("--radiomics", type=ast.literal_eval, default=False)
    parser.add_argument("--output", type=ast.literal_eval, default=False)
    parser.add_argument("--save", type=ast.literal_eval, default=False)
    
    parser.add_argument("--pca", type=int, default=32)
    parser.add_argument("--n_estimators", type=int, default=475)
    parser.add_argument("--max_leaf_nodes", type=int, default=561)
    parser.add_argument("--max_depth", type=int, default=9)
    
    args = parser.parse_args()
    return args
    
############################################################################################

if __name__ == '__main__':
    args = build_args()
    assert args.label in ['sex', 'age', 'bmi', 'weight', 'height', 'volume', 'organ', "VAT", "ASAT"]
    organs = ["liver_mesh.ply", "spleen_mesh.ply", "left_kidney_mesh.ply", "right_kidney_mesh.ply", "pancreas_mesh.ply"]
    radiomics_organs = {"liver": "liv.npz", "pancreas": "pnc.npz", "spleen": "spl.npz", "left_kidney": "lkd.npz", "right_kidney" : "rkd.npz"}
    errors=[]

    SEED = 42

    #Data
    print(f'Organ: {args.organ}, Label: {args.label}', flush=True)
    print("Data Processing", flush=True)
    # edge_path = os.path.join(args.edge_path, args.organ)
    if(args.body or args.combine_organs or args.combine_all):
        vertices_path = args.vertices_path
    else:
        vertices_path = os.path.join(args.vertices_path, args.organ)
    
    dirs = np.loadtxt("../data/NonNa_organs_split_test.txt", delimiter=",", dtype=str)
    # dirs = next(os.walk(vertices_path))[2]
    female_ids = np.loadtxt("../data/female_mesh_ids.csv")
    vol_dirs = pd.read_csv("../data/ukbb_vol_dirs.csv")
    organ_dirs = np.loadtxt("../data/NonNa_organs_split_test.txt", delimiter=",", dtype=str)
    basic_features = pd.read_csv("../data/basic_features.csv")
    basic_features_new_names = {'21003-2.0':'age', '31-0.0':'sex', '21001-2.0':'bmi', '21002-2.0':'weight','50-2.0':'height'}
    basic_features = basic_features.rename(index=str, columns=basic_features_new_names)
    body_fields = ["eid", "22407-2.0", "22408-2.0", "21080-2.0", "21081-2.0", "21082-2.0", "21083-2.0", "21084-2.0", "21087-2.0"]
    full_ukbb_data = pd.read_csv("../../../../../../vol/aimspace/projects/ukbb/668815/ukb668815_imaging.csv", usecols=body_fields)
    full_ukbb_data_new_names = {"22407-2.0": "VAT", "22408-2.0": "ASAT", '21080-2.0':'liver', '21081-2.0':'left_kidney', '21082-2.0':'right_kidney', '21083-2.0':'spleen','21087-2.0':'pancreas'}
    full_ukbb_data = full_ukbb_data.rename(index=str, columns=full_ukbb_data_new_names)
    features = pd.merge(basic_features, full_ukbb_data, how='inner') 
    # x_edges = []
    x_vertices = []
    y = []
    padded_x = []
    radiomics_nans = []
    errors = []

    if(args.combine_organs or args.combine_all):
        if(args.label == "volume"):
            args.label = args.organ

        vertices_pca = PCA(n_components=args.pca)
        for dir in dirs:
            try:
                if(not args.radiomics):
                    with open(f'{vertices_path}{dir}', "rb") as fp:
                        vertices_latent_space = pkl.load(fp)
                    fp.close()
                else: #needs to be edited to combine radiomics
                    data_path = f'{args.all_and_organ_radiomics_path}{str(dir)}'
                    if(os.path.exists(data_path)):
                        with open(data_path, "rb") as fp:
                            #all_and_organ_radiomics 
                            vertices_latent_space = pkl.load(fp)
                        fp.close()
                    else:
                        vertices_latent_space = [0,0]
                        radiomics_nans.append(dir)


                cur_patient_features = features[features['eid'] == int(dir)]
                if((not pd.isnull(cur_patient_features[args.label].item())) and not (len(vertices_latent_space) == 2)):
                    x_vertices.append(vertices_latent_space) 
                    y.append(cur_patient_features[args.label].item())
            except:
                errors.append(dir)
        # print(errors)
    elif(args.label == "organ"):
        vertices_pca = PCA(n_components=args.pca)
        for organ in organs:
            for dir in organ_dirs:
                if(not args.radiomics):
                    vertices_path = os.path.join(args.vertices_path, organ[:-9])
                    with open(f'{vertices_path}/{dir}', "rb") as fp:
                        vertices_latent_space = pkl.load(fp)
                    fp.close()
                else:
                    data_path = os.path.join(args.radiomics_path, str(dir), radiomics_organs[args.organ])
                    if(os.path.exists(data_path)):
                        data = np.load(data_path)
                        lst = data.files
                        for item in lst:
                            vertices_latent_space = data[item].flatten()
                    else:
                        vertices_latent_space = [0,0]
                        radiomics_nans.append(dir)


                if(np.asarray(vertices_latent_space).shape[0] < 61):
                    # print(f'organ: {args.organ}, dir: {dir}')
                    blah = 0
                else:
                    x_vertices.append(vertices_latent_space) 
                    y.append(organ)
    elif(args.label == "volume"):
        volume_fields = ["eid", "21080-2.0", "21081-2.0", "21082-2.0", "21083-2.0", "21084-2.0", "21087-2.0"]
        full_ukbb_data = pd.read_csv("../../../../../../vol/aimspace/projects/ukbb/668815/ukb668815_imaging.csv", usecols=volume_fields)
        new_names = {'21080-2.0':'liver', '21081-2.0':'left_kidney', '21082-2.0':'right_kidney', '21083-2.0':'spleen','21087-2.0':'pancreas'}
        full_ukbb_data = full_ukbb_data.rename(index=str, columns=new_names)
        vertices_pca = PCA(n_components=args.pca) 
        # edges_pca = PCA(n_components=args.pca) 
        for dir in np.asarray(vol_dirs):
            dir = dir[0]
            if(not args.radiomics):
                with open(f'{vertices_path}/{dir}', "rb") as fp:
                    vertices_latent_space = pkl.load(fp)
                fp.close()
                # with open(f'{edge_path}/{dir}', "rb") as fp:
                #     edge_latent_space = pkl.load(fp)
                # fp.close()
            else:
                data_path = os.path.join(args.radiomics_path, str(dir), radiomics_organs[args.organ])
                if(os.path.exists(data_path)):
                    data = np.load(data_path)
                    lst = data.files
                    for item in lst:
                        vertices_latent_space = data[item].flatten()
                else:
                    vertices_latent_space = [0,0]
                    radiomics_nans.append(dir)

            if(np.asarray(vertices_latent_space).shape[0] < 61):
                # print(f'organ: {args.organ}, dir: {dir}, shape: {np.asarray(vertices_latent_space).shape}')
                blah = 0
            else:
                x_vertices.append(vertices_latent_space) 
                # x_edges.append(edge_latent_space)
                cur_patient_features = full_ukbb_data[full_ukbb_data['eid'] == int(dir)]
                y.append(cur_patient_features[args.organ].item())
                # if(int(dir) in female_ids):
                    # y.append(0)
                # else:
                    # y.append(1)
    elif(args.body):
        vertices_pca = PCA(n_components=args.pca) 
        # edges_pca = PCA(n_componentsargs.pca=) 
        for dir in np.asarray(dirs):
            if(not args.radiomics):
                if(os.path.exists(f'{vertices_path}/{dir}.ply')):
                    with open(f'{vertices_path}/{dir}.ply', "rb") as fp:
                        vertices_latent_space = pkl.load(fp)
                    fp.close()
                    # with open(f'{edge_path}/{dir}', "rb") as fp:
                    #     edge_latent_space = pkl.load(fp)
                    # fp.close()

                    if(np.asarray(vertices_latent_space).shape[0] < args.pca):
                        # print(f'organ: {args.organ}, dir: {dir}, shape: {np.asarray(vertices_latent_space).shape}')
                        blah = 0
                    else:
                        try:
                            # x_edges.append(edge_latent_space)
                            cur_patient_features = features[features['eid'] == int(dir)]
                            if(not pd.isnull(cur_patient_features[args.label].item())):
                                x_vertices.append(vertices_latent_space) 
                                y.append(cur_patient_features[args.label].item())
                            else:
                                errors.append(dir)
                        except:
                            errors.append(dir)
                else:
                    errors.append(dir)
            else:
                data_path = f'{args.body_radiomics_path}{str(dir)}.npz'
                if(os.path.exists(data_path)):
                    data = np.load(data_path, allow_pickle=True)
                    lst = data.files
                    for item in lst:
                        vertices_latent_space = data[item]

                    if(len(vertices_latent_space) < args.pca):
                        # print(f'organ: {args.organ}, dir: {dir}, shape: {np.asarray(vertices_latent_space).shape}')
                        blah = 0
                    else:
                        try:
                            # x_edges.append(edge_latent_space)
                            cur_patient_features = features[features['eid'] == int(dir)]
                            if(not pd.isnull(cur_patient_features[args.label].item())):
                                x_vertices.append(vertices_latent_space) 
                                y.append(cur_patient_features[args.label].item())
                            else:
                                errors.append(dir)
                        except:
                            errors.append(dir)
                else:
                    vertices_latent_space = [0,0]
                    radiomics_nans.append(dir)

    elif(args.body and args.radiomics and not (args.label=="VAT" or args.label=="ASAT" or args.label=="sex")):
        vertices_pca = PCA(n_components=args.pca) 
        for dir in dirs:
            data_path = f'{args.body_radiomics_path}{str(dir)}.npz'
            if(os.path.exists(data_path)):
                data = np.load(data_path, allow_pickle=True)
                lst = data.files
                for item in lst:
                    vertices_latent_space = data[item]
            else:
                vertices_latent_space = [0,0]
                radiomics_nans.append(dir)

            try:
                cur_patient_features = basic_features[basic_features['eid'] == int(dir)]
                cur_patient_label = cur_patient_features[args.label].item()

                if((len(vertices_latent_space) < args.pca) or pd.isna(cur_patient_label) ):
                    # print(f'organ: {args.organ}, dir: {dir}')
                    blah = 0
                else:
                    x_vertices.append(vertices_latent_space) 
                    # x_edges.append(edge_latent_space)

                    y.append(cur_patient_label)
            except:
                blah = 0

    elif(not args.body and (args.label=="VAT" or args.label=="ASAT" or args.label=="sex")):
        body_fields = ["eid", "22407-2.0", "22408-2.0", "31-0.0"]
        full_ukbb_data = pd.read_csv("../../../../../../vol/aimspace/projects/ukbb/668815/ukb668815_imaging.csv", usecols=body_fields)
        label = {"VAT": "22407-2.0", "ASAT": "22408-2.0", "sex": "31-0.0"}
        vertices_pca = PCA(n_components=args.pca) 
        # edges_pca = PCA(n_components=args.pca) 
        for dir in np.asarray(dirs):
            if(not args.radiomics):
                if(os.path.exists(f'{vertices_path}/{dir}')):
                    with open(f'{vertices_path}/{dir}', "rb") as fp:
                        vertices_latent_space = pkl.load(fp)
                    fp.close()
                    # with open(f'{edge_path}/{dir}', "rb") as fp:
                    #     edge_latent_space = pkl.load(fp)
                    # fp.close()

                    if(np.asarray(vertices_latent_space).shape[0] < args.pca):
                        # print(f'organ: {args.organ}, dir: {dir}, shape: {np.asarray(vertices_latent_space).shape}')
                        blah = 0
                    else:
                        try:
                            # x_edges.append(edge_latent_space)
                            cur_patient_features = full_ukbb_data[full_ukbb_data['eid'] == int(dir)]
                            if(not pd.isnull(cur_patient_features[label[args.label]].item())):
                                x_vertices.append(vertices_latent_space) 
                                y.append(cur_patient_features[label[args.label]].item())
                            else:
                                errors.append(dir)
                        except:
                            errors.append(dir)
                else:
                    errors.append(dir)
            else:
                data_path = f'{args.radiomics_path}{str(dir)}/{radiomics_organs[args.organ]}'
                if(os.path.exists(data_path)):
                    data = np.load(data_path)
                    lst = data.files
                    for item in lst:
                        vertices_latent_space = data[item].flatten()
                    
                    if(np.asarray(vertices_latent_space).shape[0] < args.pca):
                        # print(f'organ: {args.organ}, dir: {dir}, shape: {np.asarray(vertices_latent_space).shape}')
                        blah = 0
                    else:
                        try:
                            # x_edges.append(edge_latent_space)
                            cur_patient_features = full_ukbb_data[full_ukbb_data['eid'] == int(dir)]
                            if(not pd.isnull(cur_patient_features[label[args.label]].item())):
                                x_vertices.append(vertices_latent_space) 
                                y.append(cur_patient_features[label[args.label]].item())
                            else:
                                errors.append(dir)
                        except:
                            errors.append(dir)
                else:
                    vertices_latent_space = [0,0]
                    radiomics_nans.append(dir)

    elif(not args.body and not (args.label=="VAT" or args.label=="ASAT" or args.label=="sex")): 
        vertices_pca = PCA(n_components=args.pca) 
        # edges_pca = PCA(n_components=args.pca) 
        for dir in np.asarray(dirs):
            if(not args.radiomics):
                if(os.path.exists(f'{vertices_path}/{dir}')):
                    with open(f'{vertices_path}/{dir}', "rb") as fp:
                        vertices_latent_space = pkl.load(fp)
                    fp.close()
                else:
                    errors.append(dir)
                # with open(f'{edge_path}/{dir}', "rb") as fp:
                #     edge_latent_space = pkl.load(fp)
                # fp.close()
            else:
                data_path = os.path.join(args.radiomics_path, str(dir), radiomics_organs[args.organ])
                if(os.path.exists(data_path)):
                    data = np.load(data_path)
                    lst = data.files
                    for item in lst:
                        vertices_latent_space = data[item].flatten()
                else:
                    vertices_latent_space = [0,0]
                    radiomics_nans.append(dir)

            try:
                cur_patient_features = basic_features[basic_features['eid'] == int(dir)]
                cur_patient_label = cur_patient_features[args.label].item()

                if((np.asarray(vertices_latent_space).shape[0] < args.pca) or pd.isna(cur_patient_label) ):
                    # print(f'organ: {args.organ}, dir: {dir}')
                    blah = 0
                else:
                    x_vertices.append(vertices_latent_space) 
                    # x_edges.append(edge_latent_space)

                    y.append(cur_patient_label)
            except:
                blah = 0
                
    # max_vertices = max(len(inner_array) for inner_array in x)
    if(not args.radiomics and not (args.pca == 0)):
        print("Starting PCA", flush=True)
        for i, space in enumerate(x_vertices):
            # needed_padding = max_vertices - len(space)
            # space = np.pad(space, ((0, needed_padding), (0, 0)), 'constant')

            X_vertices_pca = vertices_pca.fit_transform(x_vertices[i].T)
            # X_edges_pca = edges_pca.fit_transform(x_edges[i].T)

            X_vertices_pca = X_vertices_pca.flatten()
            # X_edges_pca = X_edges_pca.flatten()

            space = X_vertices_pca #np.append(X_vertices_pca, X_edges_pca)
            # space = space.flatten()
            padded_x.append(space)
    else:
        padded_x = x_vertices

    print(len(padded_x))
    print(len(y))
    # unique, counts = np.unique(y, return_counts=True)
    # print(dict(zip(unique, counts)))

    x_train, x_test, y_train, y_test = train_test_split(padded_x, y, test_size=0.2, random_state=SEED)

    # #Model
    print("Preparing Model", flush=True)
    parameter_space = {
        'hidden_layer_sizes': [(10), (10,20,30,10), (1,5,10,20,30), (10,10,10), (10,20,30,20,10),(6,5)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'learning_rate_init': [0.0001, 0.001, 0.01, 0.1]
    }

    if(args.label == "sex" or args.label == "organ"):
        clf = MLPClassifier(max_iter=500)
        random_search = RandomizedSearchCV(clf, parameter_space, n_iter=100, cv=5, verbose=0, random_state=42, scoring='accuracy')
    else:
        clf = MLPRegressor(max_iter=500)
        random_search = RandomizedSearchCV(clf, parameter_space, n_iter=100, cv=5, verbose=0, random_state=SEED, scoring='r2')
    
    #Training
    print("Starting Training", flush=True)
    random_search.fit(x_train, y_train)
    print('Best parameters found:\n', random_search.best_params_)

    # Predictions
    print("Getting Predictions", flush=True)
    y_pred = random_search.best_estimator_.predict(x_test)
    print(args.label)

    print("Calculating Performances", flush=True)
    if(args.label == "sex" or args.label == "organ"):
        accuracy = accuracy_score(y_test, y_pred)
        conf_mx = confusion_matrix(y_test, y_pred)
        print(f'Accuracy: {accuracy}, Confusion Matrix: {conf_mx}')
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'MSE: {mse}, R2: {r2}')
    print("---------------------------------------------", flush=True)
    
