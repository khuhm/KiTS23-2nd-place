import numpy as np
import os
import nibabel as nib
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from skimage.segmentation import watershed, expand_labels
from skimage.feature import peak_local_max
from skimage.morphology import convex_hull_image, convex_hull_object, binary_dilation, ball
from skimage.transform import hough_ellipse
from skimage import morphology
from time import time
from datetime import datetime
import logging
import itertools
from multiprocessing import Pool
from skimage.segmentation import join_segmentations
KIDNEY_MIN_AREA = 10000
KIDNEY_TOO_LARGE_AREA = 1000000
TUMOR_MIN_AREA = 100
OVERLAP_MIN_DICE = 0.3
RESULT_FOLDER = 'post_processed'
default_num_processes = 1


def postprocess_main(case, input_low, input_full, out_dir, res_folder):
    # if case[:10] in ['case_00638', 'case_00657']:  # 455, 417, 455, 436
    #     return
    print(f'*' * 80)
    print(f'{datetime.fromtimestamp(time())}\t{case[:10]}')
    seg_nib_l = nib.load(os.path.join(input_low, case))
    seg_nib_f = nib.load(os.path.join(input_full, case))
    seg_l = seg_nib_l.get_fdata()
    seg_f = seg_nib_f.get_fdata()
    spacing_l = np.fliplr(np.abs(seg_nib_l.affine[:3, :3])).diagonal()[::-1]
    spacing_f = np.fliplr(np.abs(seg_nib_f.affine[:3, :3])).diagonal()[::-1]
    scale_spacing = spacing_l[0] * spacing_l[1] * spacing_l[2]
    print(f'spacing:\t( {spacing_l[0]:.2f} x {spacing_l[1]:.2f} x {spacing_l[2]:.2f} )')
    assert np.all(np.equal(spacing_l, spacing_f)), f'spacing_l and spacing_f do not match'
# --------------------------------------------------- Masses -----------------------------------------------------------
    labeled_mass_seg_l = label(seg_l > 1)
    labeled_mass_seg_f = label(seg_f > 1)
    region_prop_mass_l = sorted(regionprops(labeled_mass_seg_l), key=lambda x: x.area, reverse=True)
    region_prop_mass_f = sorted(regionprops(labeled_mass_seg_f), key=lambda x: x.area, reverse=True)
    print(f'{"low":>15}{"Mass candidates":^50}{"full":<15}')
    for region_l, region_f in itertools.zip_longest(region_prop_mass_l, region_prop_mass_f):
        if region_l is None:
            print(f'\t' * 5, end='\t')
        else:
            print(f'{region_l.area * scale_spacing:>8,.0f}\t({region_l.centroid[0]:>3,.0f}, {region_l.centroid[1]:>3,.0f}, {region_l.centroid[2]:>3,.0f})', end='\t' * 3)
        if region_f is None:
            print()
        else:
            print(f'{region_f.area * scale_spacing:>8,.0f}\t({region_f.centroid[0]:>3,.0f}, {region_f.centroid[1]:>3,.0f}, {region_f.centroid[2]:>3,.0f})')

    print(f'{"lowres masses coincide with fullres masses":^80}')
    mass_region_l = []
    mass_region_f = []
    for region_l in region_prop_mass_l:
        region_l_coords = set([tuple(coords) for coords in region_l.coords])
        for region_f in region_prop_mass_f:
            region_f_coords = set([tuple(coords) for coords in region_f.coords])
            intersect = region_l_coords.intersection(region_f_coords)
            if not intersect:
                continue
            mass_region_l.append(region_l)
            mass_region_f.append(region_f)
            print(f'{region_l.area * scale_spacing:>8,.0f}\t({region_l.centroid[0]:>3,.0f}, {region_l.centroid[1]:>3,.0f}, {region_l.centroid[2]:>3,.0f})', end=f'\t{"<--->":^15}\t')
            print(f'{region_f.area * scale_spacing:>8,.0f}\t({region_f.centroid[0]:>3,.0f}, {region_f.centroid[1]:>3,.0f}, {region_f.centroid[2]:>3,.0f})')

    print(f'{len(mass_region_l):^80}')
    # assert len(mass_region_l) in [0], f'region_mass_l wrong number. {len(region_mass_l)} masses are detected!'

    # filtered_seg_f = np.zeros_like(seg_f)
    # for region_l in mass_region_l:
    #     filtered_seg_f[tuple(region_l.coords.transpose())] = 3
    # for region_f in mass_region_f:
    #     filtered_seg_f[tuple(region_f.coords.transpose())] = 3
    # nib.save(nib.Nifti1Image(filtered_seg_f.astype(np.uint8), seg_nib_f.affine), os.path.join(input_full, f'pp_mass', f'{case}'))

    # filtered_seg_f = np.zeros_like(seg_f)
    # filtered_seg_f[seg_f > 0] = 1
    # filtered_seg_f[seg_l > 1] = 3
    # filtered_seg_f[seg_f > 1] = 3
    # nib.save(nib.Nifti1Image(filtered_seg_f.astype(np.uint8), seg_nib_f.affine), os.path.join(input_full, f'pp_mass_union', f'{case}'))

# --------------------------------------------------- Tumor -----------------------------------------------------------
    labeled_tumor_seg_l = label(seg_l == 2)
    labeled_tumor_seg_f = label(seg_f == 2)
    region_prop_tumor_l = sorted(regionprops(labeled_tumor_seg_l), key=lambda x: x.area, reverse=True)
    region_prop_tumor_f = sorted(regionprops(labeled_tumor_seg_f), key=lambda x: x.area, reverse=True)
    print(f'{"low":>15}{"raw tumor candidates":^50}{"full":<15}')
    for region_l, region_f in itertools.zip_longest(region_prop_tumor_l, region_prop_tumor_f):
        if region_l is None:
            print(f'\t' * 5, end='\t')
        else:
            print(f'{region_l.area * scale_spacing:>8,.0f}\t({region_l.centroid[0]:>3,.0f}, {region_l.centroid[1]:>3,.0f}, {region_l.centroid[2]:>3,.0f})', end='\t' * 3)
        if region_f is None:
            print()
        else:
            print(f'{region_f.area * scale_spacing:>8,.0f}\t({region_f.centroid[0]:>3,.0f}, {region_f.centroid[1]:>3,.0f}, {region_f.centroid[2]:>3,.0f})')
    raw_region_prop_tumor_l = region_prop_tumor_l.copy()
    raw_region_prop_tumor_f = region_prop_tumor_f.copy()

    print(f'{"low":>15}{"dilate tumor regions and make convex hulls":^50}{"full":<15}')
    footprint = ball(1)
    dilated_labeled_tumor_seg_l = label(binary_dilation(seg_l == 2, footprint))
    dilated_labeled_tumor_seg_f = label(binary_dilation(seg_f == 2, footprint))
    dilated_region_prop_tumor_l = sorted(regionprops(dilated_labeled_tumor_seg_l), key=lambda x: x.area, reverse=True)
    dilated_region_prop_tumor_f = sorted(regionprops(dilated_labeled_tumor_seg_f), key=lambda x: x.area, reverse=True)
    for region_l, region_f in itertools.zip_longest(dilated_region_prop_tumor_l, dilated_region_prop_tumor_f):
        if (region_l is None) or any(np.array(region_l.image.shape) < 3) or (region_l.num_pixels < 4):
            print(f'\t' * 5, end='\t')
        else:
            sliced_seg_l = seg_l[region_l.slice]
            hull_l = np.logical_and(sliced_seg_l > 1, region_l.image_convex == 1)
            sliced_seg_l[hull_l] = 2
            print(f'{region_l.area * scale_spacing:>8,.0f}{f"->":>8}{hull_l.sum() * scale_spacing:>8,.0f}', end='\t' * 3)
        if (region_f is None) or any(np.array(region_f.image.shape) < 3) or (region_f.num_pixels < 4):
            print()
        else:
            sliced_seg_f = seg_f[region_f.slice]
            hull_f = np.logical_and(sliced_seg_f > 1, region_f.image_convex == 1)
            sliced_seg_f[hull_f] = 2
            print(f'{region_f.area * scale_spacing:>8,.0f}{f"->":>8}{hull_f.sum() * scale_spacing:>8,.0f}')

    print(f'{"low":>15}{"convexed tumor candidates":^50}{"full":<15}')
    labeled_tumor_seg_l = label(seg_l == 2)
    labeled_tumor_seg_f = label(seg_f == 2)
    region_prop_tumor_l = sorted(regionprops(labeled_tumor_seg_l), key=lambda x: x.area, reverse=True)
    region_prop_tumor_f = sorted(regionprops(labeled_tumor_seg_f), key=lambda x: x.area, reverse=True)
    for region_l, region_f in itertools.zip_longest(region_prop_tumor_l, region_prop_tumor_f):
        if region_l is None:
            print(f'\t' * 5, end='\t')
        else:
            print(f'{region_l.area * scale_spacing:>8,.0f}\t({region_l.centroid[0]:>3,.0f}, {region_l.centroid[1]:>3,.0f}, {region_l.centroid[2]:>3,.0f})', end='\t' * 3)
        if region_f is None:
            print()
        else:
            print(f'{region_f.area * scale_spacing:>8,.0f}\t({region_f.centroid[0]:>3,.0f}, {region_f.centroid[1]:>3,.0f}, {region_f.centroid[2]:>3,.0f})')

    print(f'{"lowres tumors coincide with fullres tumors":^80}')
    tumor_region_l = []
    tumor_region_f = []
    small_overlap_tumor_l = []
    small_overlap_tumor_f = []
    region_dice_list = []
    for region_l in region_prop_tumor_l:
        if region_l.area * scale_spacing < 0.:
            continue
        region_l_coords = set([tuple(coords) for coords in region_l.coords])
        for region_f in region_prop_tumor_f:
            if region_f.area * scale_spacing < 0.:
                continue
            region_f_coords = set([tuple(coords) for coords in region_f.coords])
            intersect = region_l_coords.intersection(region_f_coords)
            if not intersect:
                continue
            dice = 2 * len(intersect) / (region_l.num_pixels + region_f.num_pixels)
            if dice > OVERLAP_MIN_DICE:
                tumor_region_l.append(region_l)
                tumor_region_f.append(region_f)
                region_dice_list.append(dice)
            else:
                small_overlap_tumor_l.append(region_l)
                small_overlap_tumor_f.append(region_f)
            print(f'{region_l.area * scale_spacing:>8,.0f}\t({region_l.centroid[0]:>3,.0f}, {region_l.centroid[1]:>3,.0f}, {region_l.centroid[2]:>3,.0f})', end=f'\t{"<--":>5}{dice:^3,.3f}{"-->":<5}\t')
            print(f'{region_f.area * scale_spacing:>8,.0f}\t({region_f.centroid[0]:>3,.0f}, {region_f.centroid[1]:>3,.0f}, {region_f.centroid[2]:>3,.0f})')
    '''
    if len(small_overlap_tumor_l) > 0:
        print(f'{"small overlap dice.. split tumors":^80}')
        for region_l, region_f in itertools.zip_longest(small_overlap_tumor_l, small_overlap_tumor_f):
            if region_f.area * scale_spacing < TUMOR_MIN_AREA:
                tumor_region_l.append(region_l)
                tumor_region_f.append(region_f)
                continue
            if region_l.area > region_f.area:
                num_peaks_l = 2
                num_peaks_f = 1
            else:
                num_peaks_l = 1
                num_peaks_f = 2
            region_l_coords = set([tuple(coords) for coords in region_l.coords])
            for mass_l in raw_region_prop_tumor_l:
                mass_l_coords = set([tuple(coords) for coords in mass_l.coords])
                intersect = region_l_coords.intersection(mass_l_coords)
                if not intersect:
                    continue
                dice = 2 * len(intersect) / (region_l.num_pixels + mass_l.num_pixels)
                break
            region_f_coords = set([tuple(coords) for coords in region_f.coords])
            for mass_f in raw_region_prop_tumor_f:
                mass_f_coords = set([tuple(coords) for coords in mass_f.coords])
                intersect = region_f_coords.intersection(mass_f_coords)
                if not intersect:
                    continue
                dice = 2 * len(intersect) / (region_f.num_pixels + mass_f.num_pixels)
                break
            distance = ndi.distance_transform_edt(mass_l.image, sampling=spacing_l)
            coords = peak_local_max(distance, num_peaks=num_peaks_l, labels=mass_l.image)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)
            labels_l = watershed(-distance, markers, mask=mass_l.image)

            distance = ndi.distance_transform_edt(mass_f.image, sampling=spacing_f)
            coords = peak_local_max(distance, num_peaks=num_peaks_f, labels=mass_f.image)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)
            labels_f = watershed(-distance, markers, mask=mass_f.image)

            region_prop_small_dice_tumor_l = sorted(regionprops(labels_l, offset=mass_l.bbox[:3]), key=lambda x: x.area, reverse=True)
            region_prop_small_dice_tumor_f = sorted(regionprops(labels_f, offset=mass_f.bbox[:3]), key=lambda x: x.area, reverse=True)

            is_split_successful = False
            for sub_region_l in region_prop_small_dice_tumor_l:
                sub_region_l_coords = set([tuple(coords) for coords in sub_region_l.coords])
                for sub_region_f in region_prop_small_dice_tumor_f:
                    sub_region_f_coords = set([tuple(coords) for coords in sub_region_f.coords])
                    intersect = sub_region_l_coords.intersection(sub_region_f_coords)
                    if not intersect:
                        continue
                    dice = 2 * len(intersect) / (sub_region_l.num_pixels + sub_region_f.num_pixels)
                    if dice > OVERLAP_MIN_DICE:
                        tumor_region_l.append(sub_region_l)
                        tumor_region_f.append(sub_region_f)
                        is_split_successful = True
                        print(f'{"successful split!!!":^80}')
                    print(f'{sub_region_l.area * scale_spacing:>8,.0f}\t({sub_region_l.centroid[0]:>3,.0f}, {sub_region_l.centroid[1]:>3,.0f}, {sub_region_l.centroid[2]:>3,.0f})', end=f'\t{"<--":>5}{dice:^3,.3f}{"-->":<5}\t')
                    print(f'{sub_region_f.area * scale_spacing:>8,.0f}\t({sub_region_f.centroid[0]:>3,.0f}, {sub_region_f.centroid[1]:>3,.0f}, {sub_region_f.centroid[2]:>3,.0f})')

            if not is_split_successful:
                tumor_region_l.append(region_l)
                tumor_region_f.append(region_f)
    '''
    maybe_false_tumor_l = [region for region in region_prop_tumor_l if (region.label not in [region.label for region in tumor_region_l]) and (region.label not in [region.label for region in small_overlap_tumor_l])]
    maybe_false_tumor_f = [region for region in region_prop_tumor_f if (region.label not in [region.label for region in tumor_region_f]) and (region.label not in [region.label for region in small_overlap_tumor_f])]

    if (len(maybe_false_tumor_l) > 0) or (len(maybe_false_tumor_f) > 0):
        print(f'{"check if these are tumor false positives":^80}')
        maybe_tumor_mass_region_l = []
        maybe_tumor_mass_region_f = []
        for region_l, region_f in itertools.zip_longest(maybe_false_tumor_l, maybe_false_tumor_f):
            if region_l is None:
                print(f'\t' * 5, end='\t')
            else:
                region_l_coords = set([tuple(coords) for coords in region_l.coords])
                for mass_l in region_prop_mass_l:
                    mass_l_coords = set([tuple(coords) for coords in mass_l.coords])
                    intersect = region_l_coords.intersection(mass_l_coords)
                    if intersect:
                        dice = 2 * len(intersect) / (region_l.num_pixels + mass_l.num_pixels)
                        maybe_tumor_mass_region_l.append(mass_l)
                        print(f'{region_l.area * scale_spacing:>8,.0f}{"-":>3}{dice:^2,.2f}{"->":<3}{mass_l.area * scale_spacing:>8,.0f}', end='\t' * 3)
            if region_f is None:
                print()
            else:
                region_f_coords = set([tuple(coords) for coords in region_f.coords])
                for mass_f in region_prop_mass_f:
                    mass_f_coords = set([tuple(coords) for coords in mass_f.coords])
                    intersect = region_f_coords.intersection(mass_f_coords)
                    if intersect:
                        dice = 2 * len(intersect) / (region_f.num_pixels + mass_f.num_pixels)
                        maybe_tumor_mass_region_f.append(mass_f)
                        print(f'{region_f.area * scale_spacing:>8,.0f}{"-":>3}{dice:^2,.2f}{"->":<3}{mass_f.area * scale_spacing:>8,.0f}')

        # print(f'{"are there any matched enlarged small tumor":^80}')
        # for region_l in maybe_tumor_mass_region_l:
        #     region_l_coords = set([tuple(coords) for coords in region_l.coords])
        #     for region_f in maybe_tumor_mass_region_f:
        #         region_f_coords = set([tuple(coords) for coords in region_f.coords])
        #         intersect = region_l_coords.intersection(region_f_coords)
        #         if not intersect:
        #             continue
        #         dice = 2 * len(intersect) / (region_l.num_pixels + region_f.num_pixels)
        #         if dice > OVERLAP_MIN_DICE:
        #             tumor_region_l.append(region_l)
        #             tumor_region_f.append(region_f)
        #         print(f'{region_l.area * scale_spacing:>8,.0f}\t({region_l.centroid[0]:>3,.0f}, {region_l.centroid[1]:>3,.0f}, {region_l.centroid[2]:>3,.0f})',  end=f'\t{"<--":>5}{dice:^3,.3f}{"-->":<5}\t')
        #         print(f'{region_f.area * scale_spacing:>8,.0f}\t({region_f.centroid[0]:>3,.0f}, {region_f.centroid[1]:>3,.0f}, {region_f.centroid[2]:>3,.0f})')

    if len(tumor_region_l) > 1:
        print(f'{"more than 1 tumor matches! do something?":^80}')
        # best_match_indices = list(np.argsort(region_dice_list)[::-1][:2])
        # tumor_region_l = [tumor_region_l[idx] for idx in best_match_indices]
        # tumor_region_f = [tumor_region_f[idx] for idx in best_match_indices]

    if len(tumor_region_l) == 0:
        print(f'{"merge tumors to masses":^80}')
        maybe_tumor_mass_region_l = []
        maybe_tumor_mass_region_f = []
        self_contained_tumor_mass_l = []
        self_contained_tumor_mass_f = []
        for region_l, region_f in itertools.zip_longest(region_prop_tumor_l, region_prop_tumor_f):
            if region_l is None:
                print(f'\t' * 5, end='\t')
            else:
                region_l_coords = set([tuple(coords) for coords in region_l.coords])
                for mass_l in region_prop_mass_l:
                    mass_l_coords = set([tuple(coords) for coords in mass_l.coords])
                    intersect = region_l_coords.intersection(mass_l_coords)
                    if intersect:
                        dice = 2 * len(intersect) / (region_l.num_pixels + mass_l.num_pixels)
                        if dice > OVERLAP_MIN_DICE:
                            self_contained_tumor_mass_l.append(mass_l)
                        maybe_tumor_mass_region_l.append(mass_l)
                        print(f'{region_l.area * scale_spacing:>8,.0f}{"-":>3}{dice:^2,.2f}{"->":<3}{mass_l.area * scale_spacing:>8,.0f}', end='\t' * 3)
            if region_f is None:
                print()
            else:
                region_f_coords = set([tuple(coords) for coords in region_f.coords])
                for mass_f in region_prop_mass_f:
                    mass_f_coords = set([tuple(coords) for coords in mass_f.coords])
                    intersect = region_f_coords.intersection(mass_f_coords)
                    if intersect:
                        dice = 2 * len(intersect) / (region_f.num_pixels + mass_f.num_pixels)
                        if dice > OVERLAP_MIN_DICE:
                            self_contained_tumor_mass_f.append(mass_f)
                        maybe_tumor_mass_region_f.append(mass_f)
                        print(f'{region_f.area * scale_spacing:>8,.0f}{"-":>3}{dice:^2,.2f}{"->":<3}{mass_f.area * scale_spacing:>8,.0f}')
        print(f'{"find merged mass matches":^80}')
        for region_l in maybe_tumor_mass_region_l:
            region_l_coords = set([tuple(coords) for coords in region_l.coords])
            for region_f in maybe_tumor_mass_region_f:
                region_f_coords = set([tuple(coords) for coords in region_f.coords])
                intersect = region_l_coords.intersection(region_f_coords)
                if not intersect:
                    continue
                dice = 2 * len(intersect) / (region_l.num_pixels + region_f.num_pixels)
                if dice > OVERLAP_MIN_DICE:
                    tumor_region_l.append(region_l)
                    tumor_region_f.append(region_f)
                print(f'{region_l.area * scale_spacing:>8,.0f}\t({region_l.centroid[0]:>3,.0f}, {region_l.centroid[1]:>3,.0f}, {region_l.centroid[2]:>3,.0f})', end=f'\t{"<--":>5}{dice:^3,.3f}{"-->":<5}\t')
                print(f'{region_f.area * scale_spacing:>8,.0f}\t({region_f.centroid[0]:>3,.0f}, {region_f.centroid[1]:>3,.0f}, {region_f.centroid[2]:>3,.0f})')

        if len(tumor_region_l) == 0:
            print(f'{"find self-contained tumor mass with matched masses":^80}')
            for region_l in self_contained_tumor_mass_l:
                region_l_coords = set([tuple(coords) for coords in region_l.coords])
                any_matched = False
                for mass_f in region_prop_mass_f:
                    mass_f_coords = set([tuple(coords) for coords in mass_f.coords])
                    intersect = region_l_coords.intersection(mass_f_coords)
                    if not intersect:
                        continue
                    dice = 2 * len(intersect) / (region_l.num_pixels + mass_f.num_pixels)
                    if dice > 0:
                        any_matched = True
                        tumor_region_l.append(region_l)
                        tumor_region_f.append(mass_f)
                        print(f'{region_l.area * scale_spacing:>8,.0f}\t({region_l.centroid[0]:>3,.0f}, {region_l.centroid[1]:>3,.0f}, {region_l.centroid[2]:>3,.0f})', end=f'\t{"<--":>5}{dice:^3,.3f}{"-->":<5}\t')
                        print(f'{mass_f.area * scale_spacing:>8,.0f}\t({mass_f.centroid[0]:>3,.0f}, {mass_f.centroid[1]:>3,.0f}, {mass_f.centroid[2]:>3,.0f})')
                if not any_matched:
                    maybe_tumor_mass_region_l.append(region_l)
                    print(f'{region_l.area * scale_spacing:>8,.0f}\t({region_l.centroid[0]:>3,.0f}, {region_l.centroid[1]:>3,.0f}, {region_l.centroid[2]:>3,.0f})')

            for region_f in self_contained_tumor_mass_f:
                region_f_coords = set([tuple(coords) for coords in region_f.coords])
                any_matched = False
                for mass_l in region_prop_mass_l:
                    mass_l_coords = set([tuple(coords) for coords in mass_l.coords])
                    intersect = region_f_coords.intersection(mass_l_coords)
                    if not intersect:
                        continue
                    dice = 2 * len(intersect) / (region_f.num_pixels + mass_l.num_pixels)
                    if dice > OVERLAP_MIN_DICE:
                        any_matched = True
                        tumor_region_l.append(mass_l)
                        tumor_region_f.append(region_f)
                        print(f'{mass_l.area * scale_spacing:>8,.0f}\t({mass_l.centroid[0]:>3,.0f}, {mass_l.centroid[1]:>3,.0f}, {mass_l.centroid[2]:>3,.0f})', end=f'\t{"<--":>5}{dice:^3,.3f}{"-->":<5}\t')
                        print(f'{region_f.area * scale_spacing:>8,.0f}\t({region_f.centroid[0]:>3,.0f}, {region_f.centroid[1]:>3,.0f}, {region_f.centroid[2]:>3,.0f})')
                if not any_matched:
                    maybe_tumor_mass_region_f.append(region_f)
                    print(f'{" ":<40}\t{region_f.area * scale_spacing:>8,.0f}\t({region_f.centroid[0]:>3,.0f}, {region_f.centroid[1]:>3,.0f}, {region_f.centroid[2]:>3,.0f})')

        if (len(tumor_region_l) == 0) and (len(tumor_region_f) == 0):
            print(f'{"use merged tumor masses":^80}')
            for region_l, region_f in itertools.zip_longest(maybe_tumor_mass_region_l, maybe_tumor_mass_region_f):
                if region_l is None:
                    print(f'\t' * 5, end='\t')
                else:
                    tumor_region_l.append(region_l)
                    print(f'{region_l.area * scale_spacing:>8,.0f}\t({region_l.centroid[0]:>3,.0f}, {region_l.centroid[1]:>3,.0f}, {region_l.centroid[2]:>3,.0f})', end='\t' * 3)
                if region_f is None:
                    print()
                else:
                    tumor_region_f.append(region_f)
                    print(f'{region_f.area * scale_spacing:>8,.0f}\t({region_f.centroid[0]:>3,.0f}, {region_f.centroid[1]:>3,.0f}, {region_f.centroid[2]:>3,.0f})')

    if (len(tumor_region_l) == 0) and (len(tumor_region_f) == 0):
        print(f'{"use all tumor region candidates":^80}')
        for region in region_prop_tumor_l:
            tumor_region_l.append(region)
        for region in region_prop_tumor_f:
            tumor_region_f.append(region)

    if (len(tumor_region_l) == 0) and (len(tumor_region_f) == 0):
        print(f'{"still no tumor... convert masses into tumor":^80}')
        for region_l, region_f in itertools.zip_longest(region_prop_mass_l, region_prop_mass_f):
            if region_l is None:
                print(f'\t' * 5, end='\t')
            else:
                tumor_region_l.append(region_l)
                print(f'{region_l.area * scale_spacing:>8,.0f}\t({region_l.centroid[0]:>3,.0f}, {region_l.centroid[1]:>3,.0f}, {region_l.centroid[2]:>3,.0f})', end='\t' * 3)
            if region_f is None:
                print()
            else:
                tumor_region_f.append(region_f)
                print(f'{region_f.area * scale_spacing:>8,.0f}\t({region_f.centroid[0]:>3,.0f}, {region_f.centroid[1]:>3,.0f}, {region_f.centroid[2]:>3,.0f})')

    print(f'{"check final tumor regions":^80}')
    for region_l, region_f in itertools.zip_longest(tumor_region_l, tumor_region_f):
        if region_l is None:
            print(f'\t' * 5, end='\t')
        else:
            print(f'{region_l.area * scale_spacing:>8,.0f}\t({region_l.centroid[0]:>3,.0f}, {region_l.centroid[1]:>3,.0f}, {region_l.centroid[2]:>3,.0f})',end='\t' * 3)
        if region_f is None:
            print()
        else:
            print(f'{region_f.area * scale_spacing:>8,.0f}\t({region_f.centroid[0]:>3,.0f}, {region_f.centroid[1]:>3,.0f}, {region_f.centroid[2]:>3,.0f})')

    filtered_seg_l = np.zeros_like(seg_l)
    filtered_seg_l[seg_l > 0] = 1
    filtered_seg_l[seg_l > 1] = 3
    for region_l in tumor_region_l:
        filtered_seg_l[tuple(region_l.coords.transpose())] = seg_l[tuple(region_l.coords.transpose())]
    # for region_f in tumor_region_f:
    #     filtered_seg_l[tuple(region_f.coords.transpose())] = seg_f[tuple(region_f.coords.transpose())]
    seg_l = filtered_seg_l.copy()
    nib.save(nib.Nifti1Image(seg_l.astype(np.uint8), seg_nib_l.affine),os.path.join(input_low, f'pp_tumor_{res_folder[1]}', f'{case}'))

    filtered_seg_f = np.zeros_like(seg_f)
    filtered_seg_f[seg_f > 0] = 1
    filtered_seg_f[seg_f > 1] = 3
    for region_f in tumor_region_f:
        filtered_seg_f[tuple(region_f.coords.transpose())] = 2
    nib.save(nib.Nifti1Image(filtered_seg_f.astype(np.uint8), seg_nib_f.affine), os.path.join(input_full, f'pp_tumor_{res_folder[0]}', f'{case}'))

    filtered_seg_f = np.zeros_like(seg_f)
    filtered_seg_f[seg_l > 0] = 1
    filtered_seg_f[seg_f > 0] = 1
    filtered_seg_f[seg_l > 1] = 3
    filtered_seg_f[seg_f > 1] = 3
    # filtered_seg_f[seg_l == 2] = 2
    # filtered_seg_f[seg_f == 2] = seg_f[seg_f == 2]
    for region_l in tumor_region_l:
        filtered_seg_f[tuple(region_l.coords.transpose())] = 2
    for region_f in tumor_region_f:
        filtered_seg_f[tuple(region_f.coords.transpose())] = 2
    seg_f = filtered_seg_f.copy()

    # nib.save(nib.Nifti1Image(labeled_tumor_seg_l.astype(np.uint8), seg_nib_l.affine), os.path.join(input_low, res_folder, f'{case}'))
    nib.save(nib.Nifti1Image(seg_f.astype(np.uint8), seg_nib_f.affine), os.path.join(input_full, f'pp_tumor_union_{res_folder[0]}', f'{case}'))
    return

def postpropcess(input_low, input_full, out_dir, res_folder, num_processes):
    # os.makedirs(os.path.join(out_dir, res_folder), exist_ok=True)

    # os.makedirs(os.path.join(input_full, 'pp_mass'), exist_ok=True)
    # os.makedirs(os.path.join(input_full, 'pp_mass_union'), exist_ok=True)
    res_folder = (input_low.split(os.path.sep)[-4][31:], input_full.split(os.path.sep)[-4][31:])
    os.makedirs(os.path.join(input_low, f'pp_tumor_{res_folder[1]}'), exist_ok=True)
    # os.makedirs(os.path.join(input_full, res_folder), exist_ok=True)
    os.makedirs(os.path.join(input_full, f'pp_tumor_{res_folder[0]}'), exist_ok=True)
    os.makedirs(os.path.join(input_full, f'pp_tumor_union_{res_folder[0]}'), exist_ok=True)
    cases = sorted(os.listdir(input_low))
    cases = [case for case in cases if case.endswith('.nii.gz')]
    # cases = [case for case in cases if case not in (os.listdir(os.path.join(input_low, res_folder)))]
    # cases = [case for case in cases if case not in (os.listdir(os.path.join(input_full, f'pp_tumor_union_{res_folder[0]}')))]
    start = time()
    for case in cases:
        postprocess_main(case, input_low, input_full, out_dir, res_folder)

    # p = Pool(num_processes)
    #
    # params = []
    # for c in cases:
    #     params.append(
    #         [c, input_low, input_full, out_dir, res_folder]
    #     )
    # metrics = p.starmap(postprocess_main, params)
    # # metrics = np.vstack([i[None] for i in metrics])
    # p.close()
    # p.join()
    end = time()
    print('Evaluation took %f s. Num_processes: %d' % (np.round(end - start, 2), num_processes))
    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=str,
                        default='/home/cvip/dataset/nnUNet_results/Dataset700_KiTS2023/nnUNetTrainer__nnUNetPlans__3d_lowres_residual_all/fold_all/test/pp_kidney_fullres_batch_4_all',
                        help="low_res input folder")
    parser.add_argument('-f', type=str,
                        default='/home/cvip/dataset/nnUNet_results/Dataset700_KiTS2023/nnUNetTrainer__nnUNetPlans__3d_fullres_batch_4_all/fold_all/test/pp_kidney_lowres_residual_all',
                        help="full_res input folder")
    parser.add_argument('-o', type=str,
                        default=None,
                        help="output directory")
    parser.add_argument('-r', type=str,
                        default='pp_tumor',
                        help="result folder name for each resolution")
    parser.add_argument('-w', type=int,
                        default=1,
                        help="output directory")
    args = parser.parse_args()
    print(args)
    postpropcess(args.l, args.f, args.o, args.r, args.w)
