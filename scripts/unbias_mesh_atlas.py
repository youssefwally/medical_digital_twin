#imports
import os
import argparse
import ast
import pickle
import numpy as np

import open3d as o3d
import vtk

#Initializations
organs = ["liver_mesh.ply", "spleen_mesh.ply", "left_kidney_mesh.ply", "right_kidney_mesh.ply", "pancreas_mesh.ply"]
    
############################################################################################

#Taking arguments from user
def build_args():
    parser = argparse.ArgumentParser(description='Mesh Registration')
    parser.add_argument("--mesh_atlases_path", type=str, default="../../../../../../vol/aimspace/users/wyo/mesh_atlases/2000") 
    parser.add_argument("--transf_path", type=str, default="../../../../../../vol/aimspace/users/wyo/registration_transformations/2000") 
    parser.add_argument("--save_path", type=str, default="../../../../../../vol/aimspace/users/wyo/unbiased_mesh_atlases/2000")
    parser.add_argument("--gender", type=str, default="all")
    parser.add_argument("--organ", type=str, default="all")
    parser.add_argument("--reg_bodies", type=ast.literal_eval, default=False)
    
    parser.add_argument("--avg", type=ast.literal_eval, default=True)
    parser.add_argument("--overwrite", type=ast.literal_eval, default=False)
    
    args = parser.parse_args()
    return args
    
############################################################################################

#Create transformation field
def apply_transformation_field(matrix, vertex):
    homog_vertex = np.append(vertex, 1)  # append a 1 to make it homogeneous
    new_vertex = np.dot(matrix, homog_vertex)[:3]  # apply the transformation and remove the homogeneous coordinate
    return new_vertex
    
############################################################################################

#Unbiasing mesh atlas
def get_unbias_mesh_atlas(registration_transformations_path, mesh_atlases_path, save_path, reg_bodies, average, gender, organ):
    assert organ in organs
    assert gender in ["male", "female"]

    registration_transformations_path = os.path.join(registration_transformations_path, f'{gender}_transformations')
    
    with open(registration_transformations_path, "rb") as fp:
        registration_transformations = pickle.load(fp)

    if(reg_bodies):
        mesh_atlas_path = os.path.join(mesh_atlases_path, gender, "avg", "body.ply") 
        mesh_atlas = o3d.io.read_triangle_mesh(mesh_atlas_path)
    else:
        if(average):
            mesh_atlas_path = os.path.join(mesh_atlases_path, gender, "avg", organ) 
            mesh_atlas = o3d.io.read_triangle_mesh(mesh_atlas_path)
        else:
            mesh_atlas_path = os.path.join(mesh_atlases_path, gender, "median", organ) 
            mesh_atlas = o3d.io.read_triangle_mesh(mesh_atlas_path)  

    if(reg_bodies):
        avg_transformations = {"body_mesh": []}
        avg_transformations["body_mesh"] = np.average(registration_transformations["body_mesh"], axis = 0)

        U, _, Vt = np.linalg.svd(avg_transformations["body_mesh"][:3, :3])

        avg_transformations["body_mesh"][:3, :3] = np.dot(U, Vt)

        avg_transformations_inv = {"body_mesh": []}
        avg_transformations_inv["body_mesh"] = np.linalg.inv(avg_transformations["body_mesh"])
    else:
        avg_transformations = {"liver_mesh.ply": [], "spleen_mesh.ply": [], "left_kidney_mesh.ply": [], "right_kidney_mesh.ply": [], "pancreas_mesh.ply": []}
        avg_transformations[organ] = np.average(registration_transformations[organ], axis = 0)

        U, _, Vt = np.linalg.svd(avg_transformations[organ][:3, :3])

        avg_transformations[organ][:3, :3] = np.dot(U, Vt)

        avg_transformations_inv = {"liver_mesh.ply": [], "spleen_mesh.ply": [], "left_kidney_mesh.ply": [], "right_kidney_mesh.ply": [], "pancreas_mesh.ply": []}
        avg_transformations_inv[organ] = np.linalg.inv(avg_transformations[organ])

    unbiased_vertices = []
    for vertex in mesh_atlas.vertices:
        if(reg_bodies):
            unbiased_vertex = apply_transformation_field(avg_transformations_inv["body_mesh"] , vertex)
        else:
            unbiased_vertex = apply_transformation_field(avg_transformations_inv[organ] , vertex)
        unbiased_vertices.append(unbiased_vertex)
    unbiased_vertices = np.array(unbiased_vertices)

    mesh_atlas.vertices = o3d.utility.Vector3dVector(unbiased_vertices)

    if(reg_bodies):
        if(average):
            save_path = os.path.join(save_path, gender, "avg", "body.ply").replace("\\","/")
        else:
            save_path = os.path.join(save_path, gender, "median", "body.ply").replace("\\","/")
    else:
        if(average):
            save_path = os.path.join(save_path, gender, "avg", organ).replace("\\","/")
        else:
            save_path = os.path.join(save_path, gender, "median", organ).replace("\\","/")

    o3d.io.write_triangle_mesh(save_path, mesh_atlas, write_ascii=True)
    
############################################################################################
    
if __name__ == '__main__':
    args = build_args()
    
    if(args.gender == "all"):
        if(args.reg_bodies):   
            get_unbias_mesh_atlas(args.transf_path, args.mesh_atlases_path, args.save_path, args.reg_bodies, args.avg, "female", "liver_mesh.ply")
            get_unbias_mesh_atlas(args.transf_path, args.mesh_atlases_path, args.save_path, args.reg_bodies, args.avg, "male", "liver_mesh.ply")
            print("Created unbiased mesh atlas for all genders.")
        else:
            if(args.organ == "all"):
                for organ in organs:
                    get_unbias_mesh_atlas(args.transf_path, args.mesh_atlases_path, args.save_path, args.reg_bodies, args.avg, "female", organ)
                    get_unbias_mesh_atlas(args.transf_path, args.mesh_atlases_path, args.save_path, args.reg_bodies, args.avg, "male", organ)
                    print("Created unbiased mesh atlas for all genders and all organs.")
            else:
                get_unbias_mesh_atlas(args.transf_path, args.mesh_atlases_path, args.save_path, args.reg_bodies, args.avg, "female", args.organ)
                get_unbias_mesh_atlas(args.transf_path, args.mesh_atlases_path, args.save_path, args.reg_bodies, args.avg, "male", args.organ)
                print(f'Created unbiased mesh atlas for {args.organ} for all genders.')
    else:
        if(args.reg_bodies): 
            get_unbias_mesh_atlas(args.transf_path, args.mesh_atlases_path, args.save_path, args.reg_bodies, args.avg, args.gender, args.organ)
            print(f'Created unbiased mesh atlas for {args.gender}.')
        else:
            if(args.organ == "all"):
                for organ in organs:
                    get_unbias_mesh_atlas(args.transf_path, args.mesh_atlases_path, args.save_path, args.reg_bodies, args.avg, "female", organ)
                    get_unbias_mesh_atlas(args.transf_path, args.mesh_atlases_path, args.save_path, args.reg_bodies, args.avg, "male", organ)
                    print(f'Created unbiased mesh atlas for all {args.gender} organs.')
            else:
                get_unbias_mesh_atlas(args.transf_path, args.mesh_atlases_path, args.save_path, args.reg_bodies, args.avg, args.gender, args.organ)
                print(f'Created unbiased mesh atlas for all {args.gender} {args.organ}.')
