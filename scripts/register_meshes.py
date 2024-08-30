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
    parser.add_argument("--female_ids_path", type=str, default="../data/female_mesh_ids.csv")
    parser.add_argument("--male_ids_path", type=str, default="../data/male_mesh_ids.csv")
    # parser.add_argument("--meshes_path", type=str, default="../../../../vol/aimspace/users/wyo/organ_meshes_ply")
    parser.add_argument("--meshes_path", type=str, default="../../../../../../vol/aimspace/users/wyo/organ_decimations_ply/1000")
    parser.add_argument("--transf_save_path", type=str, default="../../../../../../vol/aimspace/users/wyo/registration_transformations/1000")
    parser.add_argument("--reg_save_path", type=str, default="../../../../../../vol/aimspace/users/wyo/registered_meshes/1000")
    parser.add_argument("--target_index", type=str, default="*")
    parser.add_argument("--gender", type=str, default="all")
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

#Register Meshes
def register_meshes(target_index, gender, ids, overwrite, reg_bodies, original_meshes_path,
                    reg_save_path, registration_transformations_path):
    assert gender in ["male", "female"]
    errors = []
    targets = []
    transformations_save_path = os.path.join(registration_transformations_path, f'{gender}_transformations')
    path = os.path.join(original_meshes_path)
    save_path = os.path.join(reg_save_path)
    
    if(reg_bodies):
         o3d_target = o3d.io.read_triangle_mesh(os.path.join(path, (target_index + ".ply")))
         trimesh_target = trimesh.load_mesh(os.path.join(path, (target_index + ".ply")))

         #WRITING
         o3d.io.write_triangle_mesh(os.path.join(save_path, (target_index + ".ply")), o3d_target, print_progress=False, write_ascii=True)
         targets.append(trimesh_target) 
    else:
        for i in range(0, len(organs)):
            o3d_target = o3d.io.read_triangle_mesh(os.path.join(path, (target_index + "/"), organs[i]))
            trimesh_target = trimesh.load_mesh(os.path.join(path, (target_index + "/"), organs[i]))

            if(not os.path.exists(os.path.join(save_path, (target_index + "/")).replace("\\","/"))):
                                    os.mkdir(os.path.join(save_path, (target_index + "/")).replace("\\","/"))

            #WRITING
            o3d.io.write_triangle_mesh(os.path.join(save_path, (target_index + "/"), organs[i]), o3d_target, print_progress=False, write_ascii=True)
            targets.append(trimesh_target) 


    

    if(reg_bodies):
        dirs = next(os.walk(path))[2]
        dirs.remove((target_index + ".ply"))
    else:
        dirs = next(os.walk(path))[1]
        dirs.remove(target_index)
    
    if(reg_bodies):
         registration_transformations = {"body_mesh": []}
    else:
        registration_transformations = {"liver_mesh.ply": [], "spleen_mesh.ply": [], "left_kidney_mesh.ply": [], "right_kidney_mesh.ply": [], "pancreas_mesh.ply": []}
    
    for dir in dirs:
        reg_already_done = True
        try:
            if(reg_bodies):
                 dir = str(dir)[:-4]
            if(str(dir) in ids):
                if(reg_bodies):
                     dir = dir + ".ply"
                     if(not os.path.exists(os.path.join(save_path, (str(dir))))):
                        reg_already_done = False
                else:
                    for i in range(0, len(organs)):
                        if(not os.path.exists(os.path.join(save_path, (str(dir) + "/"), organs[i]))):
                            reg_already_done = False

                if((not reg_already_done) or (overwrite)):
                    if(reg_bodies):
                         if(os.path.exists(os.path.join(path, (str(dir))))):
                              o3d_source = o3d.io.read_triangle_mesh(os.path.join(path, (str(dir))))
                              trimesh_source = trimesh.load_mesh(os.path.join(path, (str(dir))))
                              transformation, _ = trimesh.registration.mesh_other(trimesh_source, targets[0], samples=250, scale=False, icp_first=3, icp_final=3)
                              registration_transformations["body_mesh"].append(transformation)
                              o3d_source.transform(transformation)
                              #WRITING
                              o3d.io.write_triangle_mesh(os.path.join(save_path, (str(dir))), o3d_source, print_progress=False, write_ascii=True)
                    else:
                        for i in range(0, len(organs)):
                            if(os.path.exists(os.path.join(path, (str(dir) + "/"), organs[i]))):
                                o3d_source = o3d.io.read_triangle_mesh(os.path.join(path, (str(dir) + "/"), organs[i]))
                                trimesh_source = trimesh.load_mesh(os.path.join(path, (str(dir) + "/"), organs[i]))
                                transformation, _ = trimesh.registration.mesh_other(trimesh_source, targets[i], samples=250, scale=False, icp_first=3, icp_final=3)
                                registration_transformations[organs[i]].append(transformation)
                                o3d_source.transform(transformation)
                                if(not os.path.exists(os.path.join(save_path, (str(dir) + "/")))):
                                    os.mkdir(os.path.join(save_path, (str(dir) + "/")))
                                #WRITING
                                o3d.io.write_triangle_mesh(os.path.join(save_path, (str(dir) + "/"), organs[i]), o3d_source, print_progress=False, write_ascii=True)
        except:
             errors.append(dir)
    
    #WRITING
    with open(transformations_save_path, "wb") as fp:
        pickle.dump(registration_transformations, fp)
    fp.close()

    return errors
    
############################################################################################

if __name__ == '__main__':
    args = build_args()
    
    available_gpus = GPUtil.getAvailable()
    print("Available GPUs:", available_gpus)

    gender_dict = get_gender_ids(args.female_ids_path, args.male_ids_path)
    errors = []
    
    if(args.gender == "all"):
        target_index = "3227250"
        errors = register_meshes(target_index, "female", gender_dict["female"], args.overwrite, args.reg_bodies, args.meshes_path,
                        args.reg_save_path, args.transf_save_path)
        target_index = "1853017"
        errors = register_meshes(target_index, "male", gender_dict["male"], args.overwrite, args.reg_bodies, args.meshes_path,
                        args.reg_save_path, args.transf_save_path)
        print("Registration done for all genders.")
    else:
        ids = gender_dict[args.gender]
        if(args.target_index == "*"):
            if(args.gender == "female"):
                  target_index = "3227250"
            else:
                  target_index = "1853017"

        if(args.target_index in ids):
            errors = register_meshes(args.target_index, args.gender, ids, args.overwrite, args.reg_bodies, args.meshes_path,
                            args.reg_save_path, args.transf_save_path)
            print(f'Registration done for {args.gender}.')
        else:
             print("Aborting, Target Index is not the same gender!")
    
    print(f'Done with {len(errors)} errors.')
    
    print(f'errors: {errors}')
