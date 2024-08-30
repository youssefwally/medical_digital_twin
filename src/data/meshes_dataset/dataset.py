##########################Imports##########################
import os
import shutil
import pickle
import numpy as np
import pandas as pd

import open3d as o3d
import trimesh as trimesh
import vtk
###########################################################

class MeshProcessing():
    def __init__(self, root_path, subjects_path):
        r"""Initializes  the  Mesh    Processing 
        Class.

        Args:
            root_path (string): The path to  the 
            dataset folder.
            subjects_path (string): The path for 
            the  subjects  folder  in the dataset 
            e.g original folder.
        """
        self.root_path = root_path
        self.subjects_path = subjects_path
        self.tabular_data_path = os.path.join(self.root_path, "tabular_data")
        self.subjects = next(os.walk(self.subjects_path))[1]
        self.organs = ["liver_mesh.ply", "spleen_mesh.ply", "left_kidney_mesh.ply",
                         "right_kidney_mesh.ply", "pancreas_mesh.ply"] #Add , "body_mesh.ply"?

    def __len__(self):
        r"""Returns how many subjects are there.
        """
        return len(self.subjects)

    def get_mesh_by_index(self, idx, organ):
        r"""Creates and returns an Open3d mesh.

        Args:
            idx (int): The index of the subject.
            organ (string): Which  organ  to  be 
            used e.g liver.
        """
        assert f"{organ}_mesh.ply" in self.organs

        path = os.path.join(self.subjects_path, self.subjects[idx], f"{organ}_mesh.ply").replace("\\","/")
        mesh = o3d.io.read_triangle_mesh(path)
        return mesh

    def get_mesh_by_id(self, id, organ):
        r"""Creates and returns an Open3d mesh.

        Args:
            id (int): The  id  of  the  subject.
            organ (string): Which  organ  to  be 
            used e.g liver.
        """
        assert f"{organ}_mesh.ply" in self.organs

        path = os.path.join(self.subjects_path, str(id), f"{organ}_mesh.ply").replace("\\","/")
        mesh = o3d.io.read_triangle_mesh(path)
        return mesh

    def get_mesh_visualization(self, mesh):
        r"""Opens  a  window  with   a   visual 
        representation of the given mesh.

        Args:
            mesh (open3d mesh):  Path  to  mesh
        """
        o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, mesh_show_back_face=True, window_name="Mesh Dataset")

    def check_gender_alignment(self, gender, target_index):
        r"""Checks if the given target index  has 
        the  same  gender  as  the  given  gender.

        Args:
            gender (string):   male   or   female. 
            target_index (string):    Mesh     ID.
        """
        female_ids = np.loadtxt(os.path.join(self.tabular_data_path, "female_mesh_ids.csv").replace("\\","/"),
                                 delimiter=",", dtype=str)
        male_ids = np.loadtxt(os.path.join(self.tabular_data_path, "male_mesh_ids.csv").replace("\\","/"),
                                 delimiter=",", dtype=str)
        gender_dict = {"female": female_ids, "male": male_ids}

        if(not target_index):
            if(str(gender) == "female"):
                target_index = "3227250"
            elif(str(gender) == "male"):
                target_index = "1853017"
            else:
                target_index = "1000071"

        if(gender):
            assert str(target_index) in gender_dict[gender]
        
        return gender_dict, target_index

    def apply_transformation_field(self, matrix, vertex):
        r"""Reshapes transformation field from  the 
        registration transformations.

        Args:
            matrix (numpy array):  transformations. 
            vertex (numpy array):    mesh   vertex.
        """
        homog_vertex = np.append(vertex, 1)  # append a 1 to make it homogeneous
        new_vertex = np.dot(matrix, homog_vertex)[:3]  # apply the transformation and remove the homogeneous coordinate
        return new_vertex

    def vertex_cluster_mesh(self, mesh, voxel_size_rate, save_path = "", save = False): 
        r"""Vertex   clusters  a single mesh using 
        Open3D's simplify_vertex_clustering method 
        with the SimplificationContraction.Average.

        Args:
            mesh (open3d mesh): Mesh to be  used.
            voxel_size_rate (int): How large  the
            voxel size should be in relation   to
            the mesh. The higher the smaller  the
            voxel size. 
            save_path (string, optional): The path
            to   where   the   mesh  will be saved.
            save (bool, optional): whether to save
            the mesh or not.
        """
        try:
            voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / voxel_size_rate

            vertex_clustered_mesh = mesh.simplify_vertex_clustering(voxel_size=voxel_size,
                        contraction=o3d.geometry.SimplificationContraction.Average)
                        
            if(save):
                o3d.io.write_triangle_mesh(save_path, vertex_clustered_mesh, print_progress=False, write_ascii=True)  

            return vertex_clustered_mesh  
        except:
             raise ValueError('Vertex clustering failed!')

    def vertex_cluster_meshes(self, save_path, voxel_size_rate = 24):   
        r"""Vertex   clusters  all    meshes using 
        Open3D's simplify_vertex_clustering method 
        with the SimplificationContraction.Average.

        Args:
            voxel_size_rate (int): How large  the
            voxel size should be in relation   to
            the mesh. The higher the smaller  the
            voxel size (default 24). 
            save_path (string, optional): The path
            to   where   the   mesh  will be saved.
        """ 
        errors = []

        for subject in self.subjects:
            try:
                for organ in self.organs:
                    mesh = o3d.io.read_triangle_mesh(os.path.join(self.subjects_path, str(subject), organ).replace("\\","/"))

                    voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / voxel_size_rate

                    vertex_clusted_mesh = mesh.simplify_vertex_clustering(voxel_size=voxel_size,
                                contraction=o3d.geometry.SimplificationContraction.Average)
                                
                    vertex_clusted_mesh_path = os.path.join(save_path,  str(voxel_size_rate), str(subject), organ).replace("\\","/")

                    if(not os.path.exists(os.path.join(save_path, str(voxel_size_rate)).replace("\\","/"))):
                        os.mkdir(os.path.join(save_path, str(voxel_size_rate)).replace("\\","/"))
                        
                    if(not os.path.exists(os.path.join(save_path, str(voxel_size_rate), str(subject)).replace("\\","/"))):
                        os.mkdir(os.path.join(save_path, str(voxel_size_rate), str(subject)).replace("\\","/"))
                    


                    o3d.io.write_triangle_mesh(vertex_clusted_mesh_path, vertex_clusted_mesh, print_progress=False, write_ascii=True)    
            except:
                 errors.append(subject)

        if(len(errors) == 0):
            print("Done.")
        else:
            print(f"Vertex clustering failed for: {errors}. Total number of errors: {len(errors)}")

    def register_meshes(self, registration_save_path, registration_transformations_save_path, target_index = None, gender = None):
        r"""Registers meshes  using Trimesh's  ICP
        algorithm  registration.mesh_other method.

        Args:
            registration_save_path (string): Path 
            where the registered meshes should be  
            saved.
            registration_transformations_save_path 
            (string): Path where  the registration 
            transformations   should   be   saved.
            target_index (string, optional):   The   
            mesh id which will be used as a target
            (deafult is 1000071).
            gender (string, optional):If specified
            (female or male)   then   registration 
            will be sex_based.
        """
        assert gender in ["male", "female", None]
        gender_dict, target_index = self.check_gender_alignment(gender, target_index)

        if(gender):
            subjects = gender_dict[gender]
            registration_transformations_save_path = os.path.join(registration_transformations_save_path, f'{gender}_transformations')
        else:
            subjects = self.subjects
            registration_transformations_save_path = os.path.join(registration_transformations_save_path, 'transformations')

        errors = []
        targets = []
        registration_transformations = {"liver": [], "spleen": [], "left_kidney": [], "right_kidney": [], "pancreas": []}
    

        for organ in self.organs:
            o3d_target = o3d.io.read_triangle_mesh(os.path.join(self.subjects_path, str(target_index), organ).replace("\\","/"))
            trimesh_target = trimesh.load_mesh(os.path.join(self.subjects_path, str(target_index), organ).replace("\\","/"))

            if(not os.path.exists(os.path.join(registration_save_path, str(target_index)).replace("\\","/"))):
                                    os.mkdir(os.path.join(registration_save_path, str(target_index)).replace("\\","/"))

            o3d.io.write_triangle_mesh(os.path.join(registration_save_path, str(target_index), organ).replace("\\","/"),
                                         o3d_target, print_progress=False, write_ascii=True)
            targets.append(trimesh_target)

        subjects_indices = np.where(subjects==str(target_index))
        subjects = np.delete(subjects, subjects_indices)

        for subject in subjects:
            for i, organ in enumerate(self.organs):
                try:
                    o3d_source = o3d.io.read_triangle_mesh(os.path.join(self.subjects_path, str(subject), organ).replace("\\","/"))
                    trimesh_source = trimesh.load_mesh(os.path.join(self.subjects_path, str(subject), organ).replace("\\","/"))

                    transformation, _ = trimesh.registration.mesh_other(trimesh_source, targets[i], samples=250, scale=False, icp_first=3, icp_final=3)
                    registration_transformations[organ[:-9]].append(transformation)
                    o3d_source.transform(transformation)

                    if(not os.path.exists(os.path.join(registration_save_path, str(subject)).replace("\\","/"))):
                        os.mkdir(os.path.join(registration_save_path, str(subject)).replace("\\","/"))
                        
                    o3d.io.write_triangle_mesh(os.path.join(registration_save_path, str(subject), organ).replace("\\","/"),
                                                                 o3d_source, print_progress=False, write_ascii=True)
                except:
                     errors.append(subject)
        
        with open(registration_transformations_save_path, "wb") as fp:
            pickle.dump(registration_transformations, fp)
        fp.close()

        if(len(errors) == 0):
            print("Done.")
        else:
            print(f"Registration failed for: {errors}. Total number of errors: {len(errors)}")

    def center_mesh(self, mesh, save_path='', save = False): 
        r"""Centers  a  mesh  around  the  origin 
        using Open3D's get_center and translation
        methods.

        Args:
            mesh (open3d mesh):   Mesh   to   be 
            centered.
            save_path (string, optional) :   The
            path to where the mesh will be saved.
            save (bool, optional):   whether  to 
            save the mesh or not.
        """
        try:
            center = mesh.get_center()
            mesh.translate(-center, relative=True)
                        
            if(save):
                o3d.io.write_triangle_mesh(save_path, mesh, print_progress=False, write_ascii=True)    

            return mesh
        except:
             raise ValueError('Centering failed!')

    def center_meshes(self, save_path):   
        r"""Centers all meshes around the  origin 
        using Open3D's get_center and translation
        methods.

        Args:
            save_path (string, optional) : 
            The path to where the meshes will be 
            saved.
        """ 
        errors = []

        for subject in self.subjects:
            try:
                for organ in self.organs:
                    mesh = o3d.io.read_triangle_mesh(os.path.join(self.subjects_path, str(subject), organ).replace("\\","/"))

                    center = mesh.get_center()
                    mesh.translate(-center, relative=True)
                                
                    centered_mesh_path = os.path.join(save_path, str(subject), organ).replace("\\","/")

                    if(not os.path.exists(os.path.join(save_path, str(subject)).replace("\\","/"))):
                        os.mkdir(os.path.join(save_path, str(subject)).replace("\\","/"))

                    o3d.io.write_triangle_mesh(centered_mesh_path, mesh, print_progress=False, write_ascii=True)    
            except:
                 errors.append(subject)

        if(len(errors) == 0):
            print("Done.")
        else:
            print(f"Centering failed for: {errors}. Total number of errors: {len(errors)}")
    
    def get_mesh_atlas(self, save_path, organ, gender = None, target_index = None, average = True):
        r"""Creates   a   mesh  atlas  using  the 
        library VTK. This is acheived by  getting
        all   similar   point  accross  different 
        meshes  and  averaging  them  or  getting 
        their median. 

        Args:
            save_path (string): Path  where  mesh 
            atlases     should      be      saved.
            gender (string, optional):If specified
            (female or male)   then   registration 
            will be sex_based.
            organ (string): Which  organ  to  use.
            target_index (string, optional):   The   
            mesh id which will be used as a target
            (deafult is 1000071).
            average (bool, optional):  Get  meshes
            average  or  median  (deafult is True).
        """
        assert f"{organ}_mesh.ply" in self.organs
        assert gender in ["male", "female", None]
        gender_dict, target_index = self.check_gender_alignment(gender, target_index)

        ref_mesh_path = os.path.join(self.subjects_path, target_index, f"{organ}_mesh.ply").replace("\\","/")

        if gender:
            subjects = gender_dict[gender]
        else:
            subjects = self.subjects

        # Load reference mesh
        reader1 = vtk.vtkPLYReader()
        reader1.SetFileName(ref_mesh_path)
        reader1.Update()
        reference_mesh = reader1.GetOutput()

        meshes = []
        result_points = []

        for subject in subjects:
            path = os.path.join(self.subjects_path, (str(subject) + "/"), f"{organ}_mesh.ply").replace("\\","/")
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

        if average:
            if(not os.path.exists(os.path.join(save_path, "average").replace("\\","/"))):
                            os.mkdir(os.path.join(save_path, "average").replace("\\","/"))
                            
            temp_path = os.path.join(save_path, "average").replace("\\","/")
        else:
            if(not os.path.exists(os.path.join(save_path, "median").replace("\\","/"))):
                            os.mkdir(os.path.join(save_path, "median").replace("\\","/"))

            temp_path = os.path.join(save_path, "median").replace("\\","/")

        if gender:
            if(not os.path.exists(os.path.join(temp_path, gender).replace("\\","/"))):
                            os.mkdir(os.path.join(temp_path, gender).replace("\\","/"))

            save_path = os.path.join(temp_path, gender, f"{organ}_mesh.ply").replace("\\","/")
        else:
            if(not os.path.exists(os.path.join(temp_path, "all").replace("\\","/"))):
                            os.mkdir(os.path.join(temp_path, "all").replace("\\","/"))

            save_path = os.path.join(temp_path, "all", f"{organ}_mesh.ply").replace("\\","/")
        
        o3d.io.write_triangle_mesh(save_path, avg_mesh, write_ascii=True)

        print("Done.")

    def get_unbias_mesh_atlas(self, registration_transformations_path, mesh_atlases_path, save_path, organ, gender = None):
        r"""Creates   a   mesh  atlas  using  the 
        library VTK. This is acheived by  getting
        all   similar   point  accross  different 
        meshes  and  averaging  them  or  getting 
        their median. 

        Args:
            registration_transformations_path
            (string): Path  where  the  data   is 
            saved.
            mesh_atlases_path      (string): Path 
            where the data is saved.
            save_path (string): Path  where  mesh 
            atlases     should      be      saved.
            gender (string, optional):If specified
            (female or male)   then   registration
            will be sex_based.
            organ (string): Which  organ  to  use.
        """
        assert f"{organ}_mesh.ply" in self.organs
        assert gender in ["male", "female", None]

        if gender:
            registration_transformations_path = os.path.join(registration_transformations_path, f'{gender}_transformations').replace("\\","/")
        else:    
            registration_transformations_path = os.path.join(registration_transformations_path, 'transformations').replace("\\","/")

        with open(registration_transformations_path, "rb") as fp:
            registration_transformations = pickle.load(fp)

        if gender:
            mesh_atlas_path = os.path.join(mesh_atlases_path, gender, f"{organ}_mesh.ply").replace("\\","/")
            mesh_atlas = o3d.io.read_triangle_mesh(mesh_atlas_path)  
        else:    
            mesh_atlas_path = os.path.join(mesh_atlases_path, "all", f"{organ}_mesh.ply").replace("\\","/") 
            mesh_atlas = o3d.io.read_triangle_mesh(mesh_atlas_path)  

        avg_transformations = np.average(registration_transformations[organ], axis = 0)
        U, _, Vt = np.linalg.svd(avg_transformations[:3, :3])
        avg_transformations[:3, :3] = np.dot(U, Vt)
        avg_transformations_inv = np.linalg.inv(avg_transformations)

        unbiased_vertices = []
        for vertex in mesh_atlas.vertices:
            unbiased_vertex = self.apply_transformation_field(avg_transformations_inv, vertex)
            unbiased_vertices.append(unbiased_vertex)
        unbiased_vertices = np.array(unbiased_vertices)

        mesh_atlas.vertices = o3d.utility.Vector3dVector(unbiased_vertices)

        if gender:
            if(not os.path.exists(os.path.join(save_path, gender).replace("\\","/"))):
                            os.mkdir(os.path.join(save_path, gender).replace("\\","/"))

            save_path = os.path.join(save_path, gender, f"{organ}_mesh.ply").replace("\\","/")
        else:
            if(not os.path.exists(os.path.join(save_path, "all").replace("\\","/"))):
                            os.mkdir(os.path.join(save_path, "all").replace("\\","/"))

            save_path = os.path.join(save_path, "all", f"{organ}_mesh.ply").replace("\\","/")

        o3d.io.write_triangle_mesh(save_path, mesh_atlas, write_ascii=True)

        print("Done.")

    def get_fully_proccess_meshes(self, save_path, registration_transformations_save_path, voxel_size_rate = 24, 
                                    target_index = None, gender = None, centered = False):
        if(not os.path.exists(os.path.join(save_path, f"{voxel_size_rate}_{1 if gender else 0}_{1 if centered else 0}").replace("\\","/"))):
                            os.mkdir(os.path.join(save_path, f"{voxel_size_rate}_{1 if gender else 0}_{1 if centered else 0}").replace("\\","/"))
        
        original_subjects_path = self.subjects_path
        vertex_clustered_meshes_save_path = os.path.join(save_path, str(voxel_size_rate)).replace("\\","/")

        self.vertex_cluster_meshes(save_path, voxel_size_rate)
        self.subjects_path = vertex_clustered_meshes_save_path

        save_path = os.path.join(save_path, f"{voxel_size_rate}_{1 if gender else 0}_{1 if centered else 0}").replace("\\","/")
        self.register_meshes(save_path, registration_transformations_save_path, target_index, gender)
        self.subjects_path = save_path
        
        if centered:
            self.center_meshes(save_path)

        shutil.rmtree(vertex_clustered_meshes_save_path)
        self.subjects_path = original_subjects_path

        print("Done.")