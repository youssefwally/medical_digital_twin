#Imports
import os
import argparse
import ast

import numpy as np
from radiomics import featureextractor

############################################################################################

#Taking arguments from user
def build_args():
    parser = argparse.ArgumentParser(description='Body Pyradiomics')
    parser.add_argument("--data_path", type=str, default="../../../../../../vol/aimspace/projects/ukbb/whole_body/nifti/")
    parser.add_argument("--image_type", type=str, default="wat.nii.gz")
    parser.add_argument("--mask_path", type=str, default="../../../../../../vol/aimspace/projects/ukbb/whole_body/body_mask/")
    # parser.add_argument("--mask_type", type=str, default="body_mask.nii.gz")
    parser.add_argument("--save_path", type=str, default="../../../../../../vol/aimspace/users/wyo/radiomics/body/")

    args = parser.parse_args()
    return args
    
############################################################################################

if __name__ == '__main__':
    args = build_args()
    error = []

    # dirs = next(os.walk(args.data_path))[1]
    # dirs_path = "../../../../../../vol/aimspace/users/wyo/latent_spaces/vertices_prediction/liver"
    dirs_path = "../../../../../../vol/aimspace/projects/ukbb/whole_body/organ_segmentations/features/radiomics_features_all_enhanced/"
    dirs = next(os.walk(dirs_path))[1]
    
    # Instantiate the extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    print ("Enabled features:\n\t", extractor.enabledFeatures)

    for dir in dirs:
        try:
            data_path = f'{args.data_path}{str(dir)}/{args.image_type}'
            mask_path = f'{args.mask_path}{str(dir)}/body_mask.nii.gz'
            save_path = f'{args.save_path}{str(dir)}.npz'

            radiomics = extractor.execute(data_path, mask_path)

            np.savez(save_path, radiomics)
        
        except:
            error.append(dir)

    print("PyRadiomics for Body are Done.")





