import os
import nibabel as nib
import numpy as np
import math
from time import time
from datetime import datetime
from skimage.measure import label, regionprops
from itertools import product

TUMOR_MIN_AREA = 100


def main(list_of_input_folders, output_folder, num_processes):
    os.makedirs(output_folder, exist_ok=True)
    cases = sorted(os.listdir(list_of_input_folders[0]))
    cases = [case for case in cases if case.endswith('.nii.gz')]
    # cases = [case for case in cases if case not in (os.listdir(os.path.join(output_folder)))]
    num_preds = len(list_of_input_folders)
    for case in cases:
        print(f'{datetime.fromtimestamp(time())}\t{case[:10]}')
        list_seg_nib = [nib.load(os.path.join(input_folder, case)) for input_folder in list_of_input_folders]
        list_seg = [seg_nib.get_fdata() for seg_nib in list_seg_nib]
        spacing = np.fliplr(np.abs(list_seg_nib[0].affine[:3, :3])).diagonal()[::-1]
        scale_spacing = spacing[0] * spacing[1] * spacing[2]
        # num_majority = math.ceil(len(list_of_input_folders) / 2)
        num_majority = 1
        # vote kidney
        # vote = np.zeros_like(list_seg[0])
        # seg_out = np.zeros_like(list_seg[0])
        # for seg in list_seg:
        #     vote = vote + (seg > 0)
        # seg_out[vote >= num_majority] = 1

        # find matches
        # list_of_labeled_tumor_seg = [label(seg == 2) for seg in list_seg]
        # list_of_region_prop_tumor = [sorted(regionprops(labeled_tumor_seg), key=lambda x: x.area, reverse=True) for
        #                              labeled_tumor_seg in list_of_labeled_tumor_seg]
        # list_of_large_tumor_regions = [[region for region in region_prop_tumor if region.area * scale_spacing > TUMOR_MIN_AREA] for
        #                              region_prop_tumor in list_of_region_prop_tumor]
        # [print(f'{len(region_props):<2}', end='\t') for region_props in list_of_region_prop_tumor], print()
        # [print(f'{len(region_props):<2}', end='\t') for region_props in list_of_large_tumor_regions], print()
        # assert all([len(region_props) == len(list_of_region_prop_tumor[0]) for region_props in list_of_region_prop_tumor]), f'maybe some tumor region FPs exists'

        # list_of_tumor_regions = []

        # for region_combination in product(*tuple(list_of_large_tumor_regions)):
        #     region_coords = [set([tuple(coords) for coords in region.coords]) for region in region_combination]
        #     intersect = set.intersection(*region_coords)
        #     if not intersect:
        #         continue
        #     list_of_tumor_regions.append(region_combination)
        #
        # print(f'{len(list_of_tumor_regions):<2}')

        # seg_out = np.zeros_like(list_seg[0])
        # for tumor_regions in list_of_tumor_regions:
        #     vote = np.zeros_like(list_seg[0])
        #     for region in tumor_regions:
        #         vote[region.slice] = vote[region.slice] + region.image
        #     seg_out[vote >= num_majority] = 2

        # vote tumor
        vote = np.zeros_like(list_seg[0])
        seg_out = np.zeros_like(list_seg[0])
        for seg in list_seg:
            vote = vote + (seg > 0)
        seg_out[vote >= num_majority] = 1

        vote = np.zeros_like(list_seg[0])
        # seg_out = np.zeros_like(list_seg[0])
        for seg in list_seg:
            vote = vote + (seg > 1)
        seg_out[vote >= num_majority] = 3

        vote = np.zeros_like(list_seg[0])
        # seg_out = np.zeros_like(list_seg[0])
        for seg in list_seg:
            vote = vote + (seg == 2)
        seg_out[vote >= num_majority] = 2

        # save output segmentation
        nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), list_seg_nib[0].affine), os.path.join(output_folder, f'{case}'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs='+', type=str, required=False, default=[
        # '/home/cvip/dataset/nnUNet_results/Dataset500_KiTS2023/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/validation/pp_kidney/pp_tumor_union_lowres',
        # '/home/cvip/dataset/nnUNet_results/Dataset500_KiTS2023/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/validation/pp_kidney/pp_tumor_union_lowres_residual',
        # '/home/cvip/dataset/nnUNet_results/Dataset500_KiTS2023/nnUNetTrainer__nnUNetPlans__3d_fullres_batch_4/fold_1/validation/pp_kidney/pp_tumor_union_lowres',
        # '/home/cvip/dataset/nnUNet_results/Dataset500_KiTS2023/nnUNetTrainer__nnUNetPlans__3d_fullres_batch_4/fold_1/validation/pp_kidney/pp_tumor_union_lowres_residual',
        '/home/cvip/dataset/nnUNet_results/Dataset500_KiTS2023/nnUNetTrainer__nnUNetPlans__3d_fullres_residual/fold_1/validation/pp_kidney/pp_tumor_union_lowres',
        '/home/cvip/dataset/nnUNet_results/Dataset500_KiTS2023/nnUNetTrainer__nnUNetPlans__3d_fullres_residual/fold_1/validation/pp_kidney/pp_tumor_union_lowres_residual',
    ], help='list of input folders')
    parser.add_argument('-o', type=str, required=False, default='/home/cvip/dataset/nnUNet_results/Dataset500_KiTS2023/ensemble/fold_1/vote_mass_pairsix/vote_2',
                        help='output folder')
    parser.add_argument('-np', type=int, required=False, default=1,
                        help=f"Numbers of processes used for ensembling. Default:")
    args = parser.parse_args()
    print(args)
    main(args.i, args.o, args.np)