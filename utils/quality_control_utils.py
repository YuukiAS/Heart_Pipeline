import nibabel as nib
import numpy as np
import skimage.measure
from .image_utils import *

def sa_pass_quality_control(seg_sa_name):
    """ Quality control for short-axis image segmentation """
    nim = nib.load(seg_sa_name)
    seg_sa = nim.get_fdata()
    X, Y, Z = seg_sa.shape[:3]

    # Label class in the segmentation
    label = {'LV': 1, 'Myo': 2, 'RV': 3}

    # Criterion 1: every class exists and the area is above a threshold
    # Count number of pixels in 3D
    for l_name, l in label.items():
        pixel_thres = 10
        if np.sum(seg_sa == l) < pixel_thres:
            print('{0}: The segmentation for class {1} is smaller than {2} pixels. '
                  'It does not pass the quality control.'.format(seg_sa_name, l_name, pixel_thres))
            return False

    # Criterion 2: number of slices with LV segmentations is above a threshold
    # and there is no missing segmentation in between the slices
    z_pos = []
    for z in range(Z):
        seg_z = seg_sa[:, :, z]
        endo = (seg_z == label['LV']).astype(np.uint8)
        myo = (seg_z == label['Myo']).astype(np.uint8)
        pixel_thres = 10
        if (np.sum(endo) < pixel_thres) or (np.sum(myo) < pixel_thres):
            continue
        z_pos += [z]
    n_slice = len(z_pos)
    slice_thres = 6
    if n_slice < slice_thres:
        print('{0}: The segmentation has less than {1} slices. '
              'It does not pass the quality control.'.format(seg_sa_name, slice_thres))
        return False

    if n_slice != (np.max(z_pos) - np.min(z_pos) + 1):
        print('{0}: There is missing segmentation between the slices. '
              'It does not pass the quality control.'.format(seg_sa_name))
        return False

    # Criterion 3: LV and RV exists on the mid-cavity slice
    _, _, cz = [np.mean(x) for x in np.nonzero(seg_sa == label['LV'])]
    z = int(round(cz))
    seg_z = seg_sa[:, :, z]

    endo = (seg_z == label['LV']).astype(np.uint8)
    endo = get_largest_cc(endo).astype(np.uint8)
    myo = (seg_z == label['Myo']).astype(np.uint8)
    myo = remove_small_cc(myo).astype(np.uint8)
    epi = (endo | myo).astype(np.uint8)
    epi = get_largest_cc(epi).astype(np.uint8)
    rv = (seg_z == label['RV']).astype(np.uint8)
    rv = get_largest_cc(rv).astype(np.uint8)
    pixel_thres = 10
    if np.sum(epi) < pixel_thres or np.sum(rv) < pixel_thres:
        print('{0}: Can not find LV epi or RV to determine the AHA '
              'coordinate system.'.format(seg_sa_name))
        return False
    return True

def la_pass_quality_control(seg_la_name):
    """ Quality control for long-axis image segmentation """
    nim = nib.load(seg_la_name)
    seg = nim.get_fdata()
    X, Y, Z = seg.shape[:3]
    seg_z = seg[:, :, 0]

    # Label class in the segmentation
    label = {'LV': 1, 'Myo': 2, 'RV': 3, 'LA': 4, 'RA': 5}

    for l_name, l in label.items():
        # Criterion 1: every class exists and the area is above a threshold
        pixel_thres = 10
        if np.sum(seg_z == l) < pixel_thres:
            print('{0}: The segmentation for class {1} is smaller than {2} pixels. '
                  'It does not pass the quality control.'.format(seg_la_name, l_name, pixel_thres))
            return False

    # Criterion 2: the area is above a threshold after connected component analysis
    endo = (seg_z == label['LV']).astype(np.uint8)
    endo = get_largest_cc(endo).astype(np.uint8)
    myo = (seg_z == label['Myo']).astype(np.uint8)
    myo = remove_small_cc(myo).astype(np.uint8)
    epi = (endo | myo).astype(np.uint8)
    epi = get_largest_cc(epi).astype(np.uint8)
    pixel_thres = 10
    if np.sum(endo) < pixel_thres or np.sum(myo) < pixel_thres or np.sum(epi) < pixel_thres:
        print('{0}: Can not find LV endo, myo or epi to extract the long-axis '
              'myocardial contour.'.format(seg_la_name))
        return False
    return True


def atrium_pass_quality_control(label, label_dict):
    """ Quality control for atrial volume estimation """
    for l_name, l in label_dict.items():
        # Criterion: the atrium does not disappear at any time point so that we can
        # measure the area and length.
        T = label.shape[3]
        for t in range(T):
            label_t = label[:, :, 0, t]
            area = np.sum(label_t == l)
            if area == 0:
                print('The area of {0} is 0 at time frame {1}.'.format(l_name, t))
                return False
    return True

def aorta_pass_quality_control(image, seg):
    """ Quality control for aortic segmentation """
    for l_name, l in [('AAo', 1), ('DAo', 2)]:
        # Criterion 1: the aorta does not disappear at some point.
        T = seg.shape[3]
        for t in range(T):
            seg_t = seg[:, :, :, t]
            area = np.sum(seg_t == l)
            if area == 0:
                print('The area of {0} is 0 at time frame {1}.'.format(l_name, t))
                return False

        # Criterion 2: no strong image noise, which affects the segmentation accuracy.
        image_ED = image[:, :, :, 0]
        seg_ED = seg[:, :, :, 0]
        mean_intensity_ED = image_ED[seg_ED == l].mean()
        ratio_thres = 3
        for t in range(T):
            image_t = image[:, :, :, t]
            seg_t = seg[:, :, :, t]
            max_intensity_t = np.max(image_t[seg_t == l])
            ratio = max_intensity_t / mean_intensity_ED
            if ratio >= ratio_thres:
                print('The image becomes very noisy at time frame {0}.'.format(t))
                return False

        # Criterion 3: no fragmented segmentation
        pixel_thres = 10
        for t in range(T):
            seg_t = seg[:, :, :, t]
            cc, n_cc = skimage.measure.label(seg_t == l, return_num=True)
            count_cc = 0
            for i in range(1, n_cc + 1):
                binary_cc = (cc == i)
                if np.sum(binary_cc) > pixel_thres:
                    # If this connected component has more than certain pixels, count it.
                    count_cc += 1
            if count_cc >= 2:
                print('The segmentation has at least two connected components with more than {0} pixels '
                      'at time frame {1}.'.format(pixel_thres, t))
                return False

        # Criterion 4: no abrupt change of area
        A = np.sum(seg == l, axis=(0, 1, 2))
        for t in range(T):
            ratio = A[t] / float(A[t-1])
            if ratio >= 2 or ratio <= 0.5:
                print('There is abrupt change of area at time frame {0}.'.format(t))
                return False
    return True