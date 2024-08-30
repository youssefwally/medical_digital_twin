#imports
import os
import argparse
import ast
import csv
import pickle
import numpy as np

import GPUtil

import open3d as o3d
import trimesh

#Initializations
organs = ["liver_mesh.ply", "spleen_mesh.ply", "left_kidney_mesh.ply", "right_kidney_mesh.ply", "pancreas_mesh.ply"]
    
############################################################################################

#Taking arguments from user
#local: python .\register_meshes.py --female_ids_path ../data/female_mesh_ids.csv --male_ids_path ../data/male_mesh_ids.csv --meshes_path ../../local_data/original_meshes/ --transf_save_path ../../local_data/server_test/registration_transformations/ --reg_save_path ../../local_data/server_test/registered_meshes/ 
def build_args():
    parser = argparse.ArgumentParser(description='Mesh Registration')
    parser.add_argument("--meshes_path", type=str, default="../../../../../../vol/aimspace/users/wyo/registered_meshes/2000/")
    parser.add_argument("--save_path", type=str, default="../../../../../../vol/aimspace/users/wyo/smooth_meshes/2000/")
    parser.add_argument("--lamb", type=int, default=0.5)
    
    parser.add_argument("--overwrite", type=ast.literal_eval, default=False)
    
    
    args = parser.parse_args()
    return args
    
############################################################################################

def smooth_meshes(meshes_path, save_path, lamb):
    dirs = next(os.walk(meshes_path))[1]

    for dir in dirs:
        for organ in organs:
            path = f'{meshes_path}{dir}/{organ}'
            mesh = trimesh.load(path)
            trimesh.smoothing.filter_taubin(mesh, lamb=lamb, iterations=5)
            target_path = f'{save_path}{dir}/{organ}'
            if(not os.path.exists(os.path.join(save_path, (str(dir) + "/")))):
                                    temp_target_path = f'{save_path}{dir}/'
                                    os.mkdir(temp_target_path)
            mesh.export(target_path)


############################################################################################

if __name__ == '__main__':
    args = build_args()

    smooth_meshes(args.meshes_path, args.save_path, args.lamb)
    print("done")