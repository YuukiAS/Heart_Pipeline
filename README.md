[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://yuukias.github.io/Heart_Pipeline/)

# Overview
This is a modified version of [Wenjia Bai's Pipeline](https://github.com/baiwenjia/ukbb_cardiac) to process and analyze Cardiovascular Magnetic Resonance (CMR) data provided by UK Biobank. We extend it beyond the three modalities used in the original one to all modalities provided by UK Biobank, also adding much more features that have been defined, validated and used extensively in clinical practices.

**Note:** This repository only contains the code, not the imaging data. To know more about how to access the UK Biobank imaging data, please go to the [UK Biobank Imaging Study](http://imaging.ukbiobank.ac.uk/) website. Researchers can [apply](http://www.ukbiobank.ac.uk/register-apply/) to use the UK Biobank data resource for health-related research in the public interest.

# Tutorial

## Download Necessary Files

In addition to clone this repository, model weights and third-party tools should also be downloaded to ensure the pipeline runs as expected.

Model weights for each modality have been uploaded. You may download each separately from [Google Drive 1](https://drive.google.com/drive/folders/1ZZk1t0JsygPjX7pOBnpF9Cebyo31a6cf?dmr=1&ec=wgc-drive-globalnav-goto)

Package requirements for virtual environments of the whole pipeline, nnUNet and UMamba, as well as the third party tools have also been uploaded. You may download them from [Google Drive 2](https://drive.google.com/drive/folders/1wAqs4WFz3vrm95d6xpOFNrwrLam-I2N9?dmr=1&ec=wgc-drive-globalnav-goto). **Note you should still install VTK on your own**.

## Configure the Pipeline through `config.py`

Please ensure you have access to the following essential equipment to run the pipeline smoothly and efficiently:

+ Python3. The pipeline has been tested on Python 3.9.18
+ Environment management tool such as `conda`.
+ Slurm server for job submission. The pipeline has been tested on Slurm 24.05.4.
+ GPU Access in partitions of the Slurm server.

### Step1: Configure Pipeline Folders

1. The `pipeline_dir` should be the root folder of the entire pipeline. By default, several subfolders like `code` and `data` will be created in the root folder. You can manually set the location. For example, you may want to adjust `data_dir` if the root folder cannot accommodate all the CMR data.

### Step2: Configure Raw Data Folders

1. `data_raw_dir` does not need to be located in the `pipeline_dir`; it can be placed anywhere, as long as it follows the structure below::

    ```
    ├── 6025
    ├── 6025_retest
    ├── 20205
    ├── 20205_retest
    ├── 20207
    ├── 20208
    ├── 20208_retest
    ├── 20209
    ├── 20209_retest
    ├── 20210
    ├── 20210_retest
    ├── 20211
    ├── 20211_retest
    ├── 20212
    ├── 20212_retest
    ├── 20213
    ├── 20213_retest
    ├── 20214
    ```

    This structure aligns with the order of UK Biobank data fields:

    + **6025 and 20205**: Correspond to data fields 6025 and 20205, which store rest and exercise ECG data.
    + **Other folders**: Contain CMR data with various modalities. The pipeline extracts different features from each modality.
    + There are two main instances for the CMR data in UK Biobank. 
        - The imaging visit (2014+), treated as *visit1*.
        - The first repeat imaging visit (2019+), treated as *visit2*.
        - Folders without the `_retest` suffix store visit1 data, while folders with `_retest` store visit2 data.

        ![Short Axis Visits](https://raw.githubusercontent.com/YuukiAS/ImageHost/main/heart20250114152932126.png)

        Example of files in `20209` (visit1):
        ```
        1000033_20209_2_0.zip
        1000098_20209_2_0.zip
        1000167_20209_2_0.zip
        1000205_20209_2_0.zip
        1000240_20209_2_0.zip
        1000383_20209_2_0.zip
        1000481_20209_2_0.zip
        1000497_20209_2_0.zip
        1000560_20209_2_0.zip
        1000575_20209_2_0.zip
        ```
        Example of files in `20209_retest` (visit2):
        ```
        1000240_20209_3_0.zip
        1002872_20209_3_0.zip
        1003792_20209_3_0.zip
        1005852_20209_3_0.zip
        1007932_20209_3_0.zip
        1008114_20209_3_0.zip
        1008930_20209_3_0.zip
        1010455_20209_3_0.zip
        1010871_20209_3_0.zip
        1012433_20209_3_0.zip
        ```

2. The suffix for visit2 raw data is set to `retest` by default. You can rename this suffix, but ensure to update it in `config.py`. If visit2 data is unnecessary, set `retest_suffix = None` to only use visit1 data.

3. The pipeline also requires information about individual's Body Surface Area (BSA) and central pulse pressure information to calculate certain features. **These files should be generated on our own**. 
    + `BSA_file`: Calculate using the *Du Bois and Du Bois* formula from [height](https://biobank.ctsu.ox.ac.uk/ukb/field.cgi?id=50) and [weight](https://biobank.ctsu.ox.ac.uk/ukb/field.cgi?id=21002).
    + `pressure_file`: Extract from [UK Biobank](https://biobank.ctsu.ox.ac.uk/ukb/field.cgi?id=12678). The pipeline will only make use of the visit1 information.
 
### Step3: Configure Data Modalities

1. The complete list of modalities in `config.py` is:  
   `modality = ["aortic_scout", "la", "sa", "aortic_dist", "tag", "lvot", "aortic_blood_flow", "shmolli"]`.  
   To exclude certain modalities, simply remove them from the list.  

   Example: To only use long axis, short axis, and aortic distensibility images as per Wenjia Bai's original pipeline:  `modality = ["la", "sa", "aortic_dist"]`.

2. The pipeline can extract features from ECG data provided by UK Biobank using methods from [NeuroKit](https://neuropsychology.github.io/NeuroKit/). To disable this, set `useECG = False`.


### Step4: Configure Segmentation Settings

1. Since segmentation requires substantial GPU resources, all jobs must be submitted to a Slurm server rather than run locally. Set the `partition` to the appropriate GPU-enabled partition in your Slurm server.

2. A compatible environment for feature extraction scripts is provided in `requirements.txt` on [Google Drive 2](https://drive.google.com/drive/folders/1wAqs4WFz3vrm95d6xpOFNrwrLam-I2N9?dmr=1&ec=wgc-drive-globalnav-goto). Use this as the default environment. Environment management via `conda activate` is recommended.

3. Segmentation for modalities `20212`, `20213`, and `20214` requires **nnUNet** or **UMamba**. Configure their environments based on `nnUNet-requirements.txt` and `UMamba-requirements.txt`, also available on [Google Drive 2](https://drive.google.com/drive/folders/1wAqs4WFz3vrm95d6xpOFNrwrLam-I2N9?dmr=1&ec=wgc-drive-globalnav-goto). Update paths in `model_envs` and specify your preferred model in `model_used`. The default model is `nnUNet`.

### Step5: Configure Third-Party Tools

1. Strain calculation requires **MIRTK**, as proposed by Wenjia Bai.
2. Create a `third_party` folder in the root directory (same level as `pipeline_dir`) containing the following subfolders:  

    ```
    average_3d_ffd
    config
    MIRTK
    VTK
    ```
    All except `VTK` for Linux are zipped and available on [Google Drive 2](https://drive.google.com/drive/folders/1wAqs4WFz3vrm95d6xpOFNrwrLam-I2N9?dmr=1&ec=wgc-drive-globalnav-goto). Install `VTK` manually, preferably within the `third_party` folder.
3. Set `average_3d_ffd_path`, `MIRTK_path`, and `par_config_dir` to their respective paths in the `third_party` folder.

## Run the Pipeline

The pipeline consists of four main steps, which can be executed sequentially and with ease:

1.  Data Preparation
    `step1_prepare_data_cmr`.py: This script generates a series of job scripts to be submitted to Slurm. These jobs convert raw DICOM files into a set of NIfTI files for each individual.

    (Optional) The script `step1_prepare_data_ecg.py` can be used to copy ECG XML files directly from the raw data folder, organizing them alongside the CMR NIfTI files for each individual.
2. Image Segmentation
    `step2_segment.py`: This script creates job scripts for submission to Slurm, enabling segmentation for each imaging modality using various pre-trained models. These models are trained on subsets of UK Biobank CMR data. The resulting segmentations are stored in the same folder as the input data.
3. Feature Extraction (Separate)
    `step3_extract_feature_separate.py`: This script generates job scripts for submission to Slurm to extract features from the segmented images, which also include `batECG.sh` when `useECG` is set to `True`. During this process, several subfolders are created for each individual.

    + `feature_tracking`: Contains intermediate VTK files generated during strain calculation for feature tracking.
    + `landmark`: Includes intermediate VTK files used to determine structural features.
    + `preprocess`: Contains illustration figures comparing original NIfTI images with their preprocessed versions.
    + `tag_hd5`: Stores HDF5 files generated during strain calculations for tagged MRI images.
    + `timeseries`: Includes figures and .npz files documenting time-series data and key time points, such as end-diastole (ED) and end-systole (ES).
    + `visualization`: Serves as a quality control tool, depicting segmentation and feature extraction results in figures.

    Once all Slurm jobs are completed, the extracted features are stored as CSV files in subfolders within the `features_dir` directory. An `aggregate.pbs` script in the same directory aggregates these CSV files and organizes the results in an `aggregated` folder. The aggregated data can then be used for downstream analyses.

4. Feature Extraction (Combined)
    `step4_extract_feature_combined.py`: This final step focuses on combined feature extraction. All scripts are located in `src/feature_extraction/Combined_Features/`. Currently, two main scripts are provided:
    + `eval_ventricular_atrial_feature.py`: This script requires separate features from both long-axis and short-axis images. It calculates additional features that characterize the interaction between ventricular and atrial functions.
    + `eval_native_t1_corrected.py`: This script uses features from Native T1 images to calculate corrected T1 values. Adjustments are based on the mean $R_1=1/T_1$ value to account for water content in the myocardium.

# Future Improvements (TODO)

## Individual Modalities

### 20207
- [ ] Fix: Add units
- [ ] Validate

### 20208
- [ ] Validate
- [ ] Feature: Make use of 3-chamber long axis

### 20209
- [ ] Validate
- [ ] Feature: Change in angle caused by shear (ERC, ECL)
- [ ] Feature: Thickening

### 20210
- [ ] Validate

### 20211
- [ ] Validate
- [ ] Feature: More possible approaches such as HARP

### 20212
- [ ] Validate

### 20213
- [ ] Validate
- [ ] Feature: Determine number of cusps

### 20214
- [ ] Validate

## Miscellaneous 
- [ ] Calculate reproducibility metrics
- [ ] Improve the registration for strain computation
- [ ] Improve the segmentation quality
- [ ] Improve the preprocessing quality such as bias correction
- [ ] Potential downstream analysis such as prediction or GWAS.


# Milestones

+ 2025/1/20 Meeting
    - [ ] Visit2: Validation of 80% scripts. 
    - [ ] Visit1: Raw data should be prepared. Segmentation should start.

    *We may use L-40 GPU*
+ 2025/1/27 Meeting
    - [ ] Visit2: Validation of all scripts.
    - [ ] Visit1: Raw data should be segmented. Some features should be extracted.
+ 2025/2/3 Meeting
    - [ ] Visit1: All features extracted by end of January
    - [ ] Visit1: Calculate reproducibility information
+ 2025/2/10 Meeting
    - [ ] Review paper: Draft 1/3

    *Should I prioritize draft paper or website?*
+ 2025/2/17 Meeting
    - [ ] Review paper: Draft 2/3
+ 2025/2/24 Meeting
    - [ ] Review paper: Draft 3/3
+ 2025/3/3 Meeting
    - [ ] Review Paper: Final version
    - [ ] Website & Full Paper: 20207, 20208

    *Ensure that the citation in code is consistent: `# Ref <title> <url>`*
+ 2025/3/10 Meeting
    - [ ] Website & Full Paper: 20209, 20210
+ 2025/3/17 Meeting
    - [ ] Website & Full Paper: 20211, 20212
+ 2025/3/24 Meeting
    - [ ] Website & Full Paper: 20213, 20214
+ 2025/3/31 Meeting
    - [ ] Full Paper: Draft 1/4
+ 2025/4/7 Meeting
    - [ ] Full Paper: Draft 2/4
+ 2025/4/14 Meeting
    - [ ] Full Paper: Draft 3/4
+ 2025/4/21 Meeting
    - [ ] Full Paper: Draft 4/4
+ 2025/4/28 Meeting
    - [ ] Full Paper: Final version


# References

Thanks to the code generously provided by the following papers:

1. W. Bai, et al. Automated cardiovascular magnetic resonance image analysis with fully convolutional networks. Journal of Cardiovascular Magnetic Resonance, 20:65, 2018.
2. W. Bai, et al. Recurrent neural networks for aortic image sequence segmentation with sparse annotations. Medical Image Computing and Computer Assisted Intervention (MICCAI), 2018.
3. W. Bai, et al. A population-based phenome-wide association study of cardiac and aortic structure and function. Nature Medicine, 2020.
4. S. Petersen, et al. Reference ranges for cardiac structure and function using cardiovascular magnetic resonance (CMR) in Caucasians from the UK Biobank population cohort. Journal of Cardiovascular Magnetic Resonance, 19:18, 2017.
5. Beeche, Cameron et al. “Thoracic Aortic 3-Dimensional Geometry: Effects of Aging and Genetic Determinants.” bioRxiv : the preprint server for biology 2024.05.09.593413. 19 Aug. 2024, doi:10.1101/2024.05.09.593413. Preprint.
6. Ferdian, Edward et al. “Fully Automated Myocardial Strain Estimation from Cardiovascular MRI-tagged Images Using a Deep Learning Framework in the UK Biobank.” Radiology. Cardiothoracic imaging vol. 2,1 e190032. 27 Feb. 2020, doi:10.1148/ryct.2020190032
7. Fries, Jason A., et al. “Weakly Supervised Classification of Aortic Valve Malformations Using Unlabeled Cardiac MRI Sequences.” Nature Communications, vol. 10, no. 1, July 2019, p. 3111. DOI.org (Crossref), https://doi.org/10.1038/s41467-019-11012-3.

There are many other papers related to how to extract features from CMR images, as well as the reference ranges for such features, please see our paper.