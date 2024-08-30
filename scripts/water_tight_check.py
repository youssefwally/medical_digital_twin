#imports
import os
import argparse
import ast

import open3d as o3d

#Initializations
organs = ["liver_mesh.ply", "spleen_mesh.ply", "left_kidney_mesh.ply", "right_kidney_mesh.ply", "pancreas_mesh.ply"]
    
############################################################################################

#Taking arguments from user
def build_args():
    parser = argparse.ArgumentParser(description='Mesh Creation')
    parser.add_argument("--original_meshes_path", type=str, default="../../../../../vol/aimspace/users/wyo/organ_meshes_ply/")
    parser.add_argument("--dec_meshes_path", type=str, default="../../../../../vol/aimspace/users/wyo/registered_meshes/2000/")
    parser.add_argument("--vc_meshes_path", type=str, 
                                        default="../../../../../vol/aimspace/users/wyo/organ_decimations_ply/vertex_clustering/")

    args = parser.parse_args()
    return args

############################################################################################

def get_water_tightness(original_meshes_path, dec_meshes_path, vc_meshes_path):
    dirs = next(os.walk(original_meshes_path))[1]
    organs = ["liver_mesh.ply", "spleen_mesh.ply", "left_kidney_mesh.ply", "right_kidney_mesh.ply", "pancreas_mesh.ply"]

    water_tightness = {"liver": [0, 0, 0, 0, 0, 0], "left_kidney": [0, 0, 0, 0, 0, 0], "right_kidney": [0, 0, 0, 0, 0, 0], 
                    "spleen": [0, 0, 0, 0, 0, 0], "pancreas": [0, 0, 0, 0, 0, 0]}
    output = {"True": 1, "False": 0}
    print("starting", flush=True)
    for dir in dirs:
        for organ in organs:
            original_mesh_path = f'{original_meshes_path}{str(dir)}/{organ}'
            dec_mesh_path = f'{dec_meshes_path}{str(dir)}/{organ}'
            vc_mesh_path = f'{vc_meshes_path}{str(dir)}/{organ}'

            original_mesh = o3d.io.read_triangle_mesh(original_mesh_path)
            dec_mesh = o3d.io.read_triangle_mesh(dec_mesh_path)
            vc_mesh = o3d.io.read_triangle_mesh(vc_mesh_path)

            original_watertight = original_mesh.is_watertight()
            dec_watertight = dec_mesh.is_watertight()
            vc_watertight = vc_mesh.is_watertight()

            water_tightness[str(organ[:-9])][output[str(original_watertight)]] += 1
            water_tightness[str(organ[:-9])][output[str(dec_watertight)] + 2] += 1
            water_tightness[str(organ[:-9])][output[str(vc_watertight)] + 4] += 1
    
    return water_tightness

############################################################################################

if __name__ == '__main__':
    args = build_args()
    
    water_tightness = get_water_tightness(args.original_meshes_path, args.dec_meshes_path, args.vc_meshes_path)

    print("Order: [original_not_wt, original_wt, dec_not_wt, dec_wt, vc_not_wt, vc_wt]")
    print(water_tightness)