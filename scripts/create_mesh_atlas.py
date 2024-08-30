#imports
import os
import argparse
import ast
import csv
import numpy as np

import open3d as o3d
import vtk

#Initializations
organs = ["liver_mesh.ply", "spleen_mesh.ply", "left_kidney_mesh.ply", "right_kidney_mesh.ply", "pancreas_mesh.ply"]
    
############################################################################################

#Taking arguments from user
def build_args():
    parser = argparse.ArgumentParser(description='Mesh Registration')
    parser.add_argument("--female_ids_path", type=str, default="../../../../../../vol/aimspace/projects/ukbb/abdominal/mesh_data/organ_mesh/female_mesh_ids.csv")
    parser.add_argument("--male_ids_path", type=str, default="../../../../../../vol/aimspace/projects/ukbb/abdominal/mesh_data/organ_mesh/male_mesh_ids.csv")
    parser.add_argument("--reg_meshes_path", type=str, default="../../../../../../vol/aimspace/users/wyo/registered_meshes/2000")
    parser.add_argument("--save_path", type=str, default="../../../../../../vol/aimspace/users/wyo/mesh_atlases/2000")
    parser.add_argument("--target_index", type=str, default="*")
    parser.add_argument("--gender", type=str, default="all")
    parser.add_argument("--organ", type=str, default="all")
    parser.add_argument("--reg_bodies", type=ast.literal_eval, default=False)
    
    parser.add_argument("--overwrite", type=ast.literal_eval, default=False)
    
    args = parser.parse_args()
    return args
    
############################################################################################

#Splitting ids to female and male
def get_gender_ids(female_path, male_path):
    with open(female_path, newline='') as csvfile:
        female_ids = list(csv.reader(csvfile, delimiter=","))
        female_ids = np.asarray(female_ids)
        female_ids = female_ids.flatten()
    
    with open(male_path, newline='') as csvfile:
        male_ids = list(csv.reader(csvfile, delimiter=","))
        male_ids = np.asarray(male_ids)
        male_ids = male_ids.flatten() 
    
    gender_dict = {"female": female_ids, "male": male_ids}
    
    return gender_dict 
    
############################################################################################

#Createing Mesh atlas
def get_vtk_mesh_atlas(target_index, ids, reg_bodies, registered_organs_path, save_path, gender, organ, average = True):
    assert organ in organs
    assert gender in ["male", "female"]

    registered_data_path = os.path.join(registered_organs_path)

    if(reg_bodies):
        ref_mesh_path = os.path.join(registered_organs_path, (target_index + ".ply"))

        dirs = next(os.walk(registered_data_path))[2]
    else:
        ref_mesh_path = os.path.join(registered_organs_path, target_index, organ)

        dirs = next(os.walk(registered_data_path))[1]

    # Load reference mesh
    reader1 = vtk.vtkPLYReader()
    reader1.SetFileName(ref_mesh_path)
    reader1.Update()
    reference_mesh = reader1.GetOutput()

    meshes = []
    result_points = []

    for dir in dirs:
        if(reg_bodies):
            dir = str(dir)[:-4]
        if(str(dir) in ids):
            if(reg_bodies):
                dir = dir + ".ply"
            if(reg_bodies):
                path = os.path.join(registered_data_path, (str(dir)))
            else:
                path = os.path.join(registered_data_path, (str(dir) + "/"), organ)
            # Load other meshes
            reader2 = vtk.vtkPLYReader()
            reader2.SetFileName(path)
            reader2.Update()
            mesh = reader2.GetOutput()
            meshes.append(mesh)

    # Loop over the points in reference mesh and find the closest point other meshes
    for i in range(reference_mesh.GetNumberOfPoints()):
        point = reference_mesh.GetPoint(i)
        close_points = []
        for mesh in meshes:
            # Create a point locator for mesh
            locator = vtk.vtkPointLocator()
            locator.SetDataSet(mesh)
            locator.BuildLocator()
            closest_point_id = locator.FindClosestPoint(point)
            closest_point = mesh.GetPoint(closest_point_id)
            close_points.append(closest_point)
        if(average):
            result_points.append(np.average(close_points, axis = 0))
        else:
            result_points.append(np.median(close_points, axis = 0))
            
    avg_mesh = o3d.io.read_triangle_mesh(ref_mesh_path)
    avg_mesh.vertices = o3d.utility.Vector3dVector(result_points)
    print(avg_mesh)

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
    print(save_path)
    o3d.io.write_triangle_mesh(save_path, avg_mesh, write_ascii=True)     
    
############################################################################################
    
if __name__ == '__main__':
    args = build_args()

    gender_dict = get_gender_ids(args.female_ids_path, args.male_ids_path)
    
    if(args.gender == "all"):
        if(args.reg_bodies):
            target_index = "3227250"
            ids = gender_dict["female"]
            get_vtk_mesh_atlas(target_index, ids, args.reg_bodies, args.reg_meshes_path, args.save_path, "female", "liver_mesh.ply", average = True)
            target_index = "1853017"
            ids = gender_dict["male"]
            get_vtk_mesh_atlas(target_index, ids, args.reg_bodies, args.reg_meshes_path, args.save_path, "male", "liver_mesh.ply", average = True)
            print(f'Created biased mesh atlas for all genders.')
        else:
            if(args.organ == "all"):
                for organ in organs:
                    target_index = "3227250"
                    ids = gender_dict["female"]
                    get_vtk_mesh_atlas(target_index, ids, args.reg_bodies, args.reg_meshes_path, args.save_path, "female", organ, average = True)
                    target_index = "1853017"
                    ids = gender_dict["male"]
                    get_vtk_mesh_atlas(target_index, ids, args.reg_bodies, args.reg_meshes_path, args.save_path, "male", organ, average = True)
                print("Created biased mesh atlas for all genders and all organs.")
            else:
                target_index = "3227250"
                ids = gender_dict["female"]
                get_vtk_mesh_atlas(target_index, ids, args.reg_bodies, args.reg_meshes_path, args.save_path, "female", args.organ, average = True)
                target_index = "1853017"
                ids = gender_dict["male"]
                get_vtk_mesh_atlas(target_index, ids, args.reg_bodies, args.reg_meshes_path, args.save_path, "male", args.organ, average = True)
                print(f'Created biased mesh atlas for {args.organ} for all genders.')
    else:
        ids = gender_dict[args.gender]
        if(args.target_index == "*"):
            if(args.gender == "female"):
                  target_index = "3227250"
            else:
                  target_index = "1853017"
        else:
            target_index = args.target_index

        if(target_index in ids):
            if(args.reg_bodies):
                get_vtk_mesh_atlas(target_index, ids, args.reg_bodies, args.reg_meshes_path, args.save_path, args.gender, "liver_mesh.ply", average = True)
                print(f'Created biased mesh atlas for {args.gender}.')
            else:
                if(args.organ == "all"):
                    for organ in organs:
                        get_vtk_mesh_atlas(target_index, ids, args.reg_bodies, args.reg_meshes_path, args.save_path, args.gender, organ, average = True)
                    print(f'Created biased mesh atlas for all {args.gender} organs.')
                else:
                    get_vtk_mesh_atlas(target_index, ids, args.reg_bodies, args.reg_meshes_path, args.save_path, args.gender, args.organ, average = True)
                    print(f'Created biased mesh atlas for all {args.gender} {args.organ}.')
        else:
             print("Aborting, Target Index is not the same gender!")
