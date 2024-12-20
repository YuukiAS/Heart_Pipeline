[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://yuukias.github.io/Heart_Pipeline/)

# Overview
This is a modified version of [Wenjia Bai's Pipeline](https://github.com/baiwenjia/ukbb_cardiac) to process UKBiobank data.


## Preparation

Please make sure `data_raw_dir` in `config.py` have the following structure:

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

where `retest` should be the same as `retest_suffix` field in `config.py`.

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