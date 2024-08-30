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
    parser.add_argument("--meshes_path", type=str, default="../../../../../../vol/aimspace/users/wyo/organ_meshes_ply")
    parser.add_argument("--save_path", type=str, default="../../../../../../vol/aimspace/users/wyo/organ_decimations_ply")
    parser.add_argument("--decimation_rate", type=int, default=2000)
    
    parser.add_argument("--overwrite", type=ast.literal_eval, default=False)
    
    args = parser.parse_args()
    return args

############################################################################################

def decimate_meshes(organ_meshes_path, save_path, decimation_rate, overwrite):    
    errors = []
    dirs = next(os.walk(os.path.join(organ_meshes_path)))[1]

    if(not os.path.exists(os.path.join(save_path, str(decimation_rate)).replace("\\","/"))):
        os.mkdir(os.path.join(save_path, str(decimation_rate)).replace("\\","/"))

    
    finished_meshes = next(os.walk(os.path.join(save_path, str(decimation_rate)).replace("\\","/")))[1]

    for dir in dirs:
            if((not str(dir) in finished_meshes) or not (len(next(os.walk(os.path.join(save_path, str(dir)).replace("\\","/")))[2]) == 5) or overwrite):
                try:
                    for organ in organs:
                        mesh = o3d.io.read_triangle_mesh(os.path.join(organ_meshes_path, str(dir), organ).replace("\\","/"))
                        dec_mesh = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, decimation_rate)

                        dec_mesh_path = os.path.join(save_path, str(decimation_rate), str(dir), organ).replace("\\","/")
                        if(not os.path.exists(os.path.join(save_path, str(decimation_rate), str(dir)).replace("\\","/"))):
                            os.mkdir(os.path.join(save_path, str(decimation_rate), str(dir)).replace("\\","/"))

                        o3d.io.write_triangle_mesh(dec_mesh_path, dec_mesh, print_progress=False, write_ascii=True)    
                except:
                     errors.append(dir)
    
    return errors

############################################################################################

if __name__ == '__main__':
    args = build_args()

    errors = decimate_meshes(args.meshes_path, args.save_path, args.decimation_rate, args.overwrite)
    print(f'Created decimated meshes with {len(errors)} errors.')
    
    print(f'errors: {errors}')

    