---
title: Medical Imaging Format and Computation Basics
---

## DICOM Files

### Overview

DICOM (Digital Imaging in Medicine) is the fundamental format of medical image datasets, storage and transfer. DICOM files contain headers that provide abundant information regarding the protocol, patient, and acquisition details. 

In python, `pydicom` can be used to work with DICOM files.

Most CMR data in UKBiobank are provided in DICOM format. However, each file only corresponds to a single time frame on a single slice and may be difficult to work with during segmentation/registration.

## NifTI Files


### Overview

NifTI (Neuroimaging Informatics Technology Initiative) is a standard format for neuroimaging data. NifTI files are stored in `.nii` or `.nii.gz` format.

In python, `nibabel` can be used to work with NifTI files.

In this pipeline, we convert the DICOM files to NifTI format for easier processing. The details can be found in `script/prepare_data.py` and `utils/biobank_utils.py`.


### Header and Affine Matrix

When working with NifTI files, we usually need to deal with two fields: `affine` and `header`. 

The NifTI header contains a `pixdim` field, which is frequently used and describes the voxel size in each dimension. **It doesn't account for rotation and translation, but only provides the scale of the voxels**:

+ `pixdim[1]` is the voxel size in the x-direction, usually in millimeters
+ `pixdim[2]` is the voxel size in the y-direction
+ `pixdim[3]` is the voxel size in the z-direction
+ `pixdim[4]` is the temporal resolution (if applicable), usually in seconds.

The `affine` matrix, on the other hand, encapsulates **voxel size, rotation as well as translation**.

The affine matrix takes the following form:

$$
\text{Affine} = 
\begin{bmatrix}
s_x r_{11} & s_y r_{12} & s_z r_{13} & t_x \\
s_x r_{21} & s_y r_{22} & s_z r_{23} & t_y \\
s_x r_{31} & s_y r_{32} & s_z r_{33} & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

+ $s_x, s_y, s_z$: Scaling factors that represent voxel sizes in $x,y,z$ directions
+ $r_{ij}$: Direction cosines, indicating the rotation of the voxel grid relative to the physical space
+ $t_x, t_y, t_z$: Translation factors, representing the origin of the image in physical space

Therefore, $(s_xr_{11})^2+(s_xr_{21})^2+(s_xr_{31})^2=pixdim[1]$, both representing the voxel size in x dimension.

> Example1: The short axis file of patient 100240 at visit2, the affine matrix is 
$\text{Affine} = \begin{bmatrix}
-0.5549 & 1.4533 & -5.2436 & -140.9455 \\
-1.4707 & 0.1463 & 5.8785 & 67.7619 \\
-0.9310 & -1.0974 & -6.1602 & 259.3229 \\
0 & 0 & 0 & 1
\end{bmatrix}$. The corresponding pixel dimension is $[-1.,  1.826923,  1.826923, 10.,  0.02462 ,  1.,  1.,  1.]$. Then the voxel size in x and y dimension is 1.826923mm. The slice gap + slice thickness is 100mm. The temporal resolution between time frames is 24.62ms.


> Example2: The third column of the affine matrix represents the scaling and rotation in z direction, which is the **normal vector that aligns with the direction of stack of slices**. Since we need to center perpendicular to the septum in HLA to obtain SA view, the normalized vector `long_axis = nim_sa.affine[:3, 2] / np.linalg.norm(nim_sa.affine[:3, 2])` hence represents direction of long axis.

> Example3: In the short axis we may have two set of landmarks on the same slice for the same time frame $(x_1, y_1)$ and $(x_2, y_2)$ that are used to determine the transverse diameter of the left ventricle. In this case we can multiply each landmark through `np.dot(nim_sa.affine, np.array([x, y, 0, 1]))[:3]`. Then the diameter can be directly determined through `np.linalg.norm`..` 

