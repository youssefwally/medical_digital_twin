#imports
import os
import argparse
import ast
import numpy as np

import nibabel as nib
from skimage.measure import marching_cubes 
import pyvista
import open3d as o3d

#Initializations
organs = ["liver_mesh.ply", "spleen_mesh.ply", "left_kidney_mesh.ply", "right_kidney_mesh.ply", "pancreas_mesh.ply"]
    
############################################################################################

#Taking arguments from user
def build_args():
    parser = argparse.ArgumentParser(description='Mesh Creation')
    parser.add_argument("--seg_path", type=str, default="../../../../../../vol/aimspace/projects/ukbb/abdominal/mesh_data/organ_mesh/organ_segmentations/")
    parser.add_argument("--meshes_save_path", type=str, default="../../../../../../vol/aimspace/users/wyo/organ_meshes_ply")
    
    parser.add_argument("--overwrite", type=ast.literal_eval, default=False)
    
    args = parser.parse_args()
    return args

############################################################################################

def pad_edge_list(edges):
    padding = np.ones(edges.shape[0], int)*3
    edges_w_padding = np.vstack((padding, edges.T)).T
    return edges_w_padding
    
############################################################################################

#load organ meshes data
def load_meshes(segmentations_path, patient_id):
    abdominal_segmentations = nib.load(os.path.join(segmentations_path, str(patient_id), "prd.nii.gz"))
    abdominal_segmentations_data = abdominal_segmentations.get_fdata()
    
    #1:Liver 2:Spleen 3:Left Kidney 4:Right Kidney 5:Pancreas 
    verts1, faces1, norms1, vals1 = marching_cubes(abdominal_segmentations_data==1, level=0, step_size=1)
    verts2, faces2, norms2, vals2 = marching_cubes(abdominal_segmentations_data==2, level=0, step_size=1)
    verts3, faces3, norms3, vals3 = marching_cubes(abdominal_segmentations_data==3, level=0, step_size=1)
    verts4, faces4, norms4, vals4 = marching_cubes(abdominal_segmentations_data==4, level=0, step_size=1)
    verts5, faces5, norms5, vals5 = marching_cubes(abdominal_segmentations_data==5, level=0, step_size=1)
    verts1 = verts1/np.array(abdominal_segmentations_data.shape) 
    verts2 = verts2/np.array(abdominal_segmentations_data.shape) 
    verts3 = verts3/np.array(abdominal_segmentations_data.shape) 
    verts4 = verts4/np.array(abdominal_segmentations_data.shape) # to normalize ponit coordinate in [0,1]
    verts5 = verts5/np.array(abdominal_segmentations_data.shape) # to normalize ponit coordinate in [0,1]
    edges1 = np.concatenate((faces1[:,:2], faces1[:,1:]), axis=0)
    edges2 = np.concatenate((faces2[:,:2], faces2[:,1:]), axis=0)
    edges3 = np.concatenate((faces3[:,:2], faces3[:,1:]), axis=0)
    edges4 = np.concatenate((faces4[:,:2], faces4[:,1:]), axis=0)
    edges5 = np.concatenate((faces5[:,:2], faces5[:,1:]), axis=0)

    lines1 = np.concatenate((np.int32(2*np.ones((edges1.shape[0],1))), edges1), 1)
    lines2 = np.concatenate((np.int32(2*np.ones((edges2.shape[0],1))), edges2), 1)
    lines3 = np.concatenate((np.int32(2*np.ones((edges3.shape[0],1))), edges3), 1)
    lines4 = np.concatenate((np.int32(2*np.ones((edges4.shape[0],1))), edges4), 1)
    lines5 = np.concatenate((np.int32(2*np.ones((edges5.shape[0],1))), edges5), 1)
    mesh1 = pyvista.PolyData(verts1, pad_edge_list(faces1))
    mesh2 = pyvista.PolyData(verts2, pad_edge_list(faces2))
    mesh3 = pyvista.PolyData(verts3, pad_edge_list(faces3))
    mesh4 = pyvista.PolyData(verts4, pad_edge_list(faces4))
    mesh5 = pyvista.PolyData(verts5, pad_edge_list(faces5))

    mesh1.lines = lines1.flatten()
    mesh2.lines = lines2.flatten()
    mesh3.lines = lines3.flatten()
    mesh4.lines = lines4.flatten()
    mesh5.lines = lines5.flatten()

    verts = [verts1, verts2, verts3, verts4, verts5]
    edges = [edges1, edges2, edges3, edges4, edges5]
    faces = [faces1, faces2, faces3, faces4, faces5]
    lines = [lines1, lines2, lines3, lines4, lines5]
    meshes = [mesh1, mesh2, mesh3, mesh4, mesh5]
    norms = [norms1, norms2, norms3, norms4, norms5]
    vals = [vals1, vals2, vals3, vals4, vals5]
    
    return verts, edges, faces, lines, meshes, norms, vals
    
############################################################################################

#Make Triangle Mesh from meshes
def get_triangle_mesh(verts, faces):
    mesh = o3d.geometry.TriangleMesh(vertices = o3d.utility.Vector3dVector(np.asarray(verts)),
                                     triangles=o3d.utility.Vector3iVector(np.asarray(faces)))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0, 0.3, 0.5])
    
    return mesh
    
############################################################################################

#Save Triangle Mesh
def save_triangle_mesh(triangle_mesh, patient_id, organ, save_path):

    path = os.path.join(save_path, str(patient_id))
    if(not os.path.exists(os.path.join(path))):
        os.mkdir(path)
    save_path = os.path.join(path, organ)
    
    #WRITING
    o3d.io.write_triangle_mesh(save_path, triangle_mesh)
    
############################################################################################

#Creating 3d Meshes
def create_meshes(segmentations_path, save_path, overwrite):   
    
    dirs = next(os.walk(segmentations_path))[1]
    finished_meshes = next(os.walk(save_path))[1]
    errors = []
    
    for dir in dirs:
        if((not str(dir) in finished_meshes) or not (len(next(os.walk(os.path.join(save_path, str(dir))))[2]) == 5) or overwrite):
            try:
                verts, edges, faces, lines, meshes, norms, vals = load_meshes(segmentations_path, dir)
                for i, organ in enumerate(organs):
                    mesh = get_triangle_mesh(verts[i], faces[i])
                    # dec_mesh = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, rate)
                    save_triangle_mesh(mesh, dir, organ, save_path)
            except:
                errors.append(dir)

    return errors
    
############################################################################################

if __name__ == '__main__':
    args = build_args()

    errors = create_meshes(args.seg_path, args.meshes_save_path, args.overwrite)
    print(f'Created Organ Meshes with {len(errors)} errors.')
    
    print(f'errors: {errors}')

    