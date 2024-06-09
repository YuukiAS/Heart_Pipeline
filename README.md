# Overview
This is a modified version of [Wenjia Bai's Pipeline](https://github.com/baiwenjia/ukbb_cardiac) to process UKBiobank data.

## Note on UU-Mamba

1. Clone the repo in https://github.com/tiffany9056/UU-Mamba and install the package using `pip install -e .`. 
2. As this will override the usage of `nnUNetv2`, it is recommended to create another virtual environment and specify path of the other virtual environment in `config.py`.
3. Modify `uumamba/nnunetv2/paths.py` to allow dynamic environment variables
```python
# nnUNet_raw = join(base, 'nnUNet_raw') # os.environ.get('nnUNet_raw')
# nnUNet_preprocessed = join(base, 'nnUNet_preprocessed') # os.environ.get('nnUNet_preprocessed')
# nnUNet_results = join(base, 'nnUNet_results') # os.environ.get('nnUNet_results')
nnUNet_raw = os.environ.get('nnUNet_raw')
nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
nnUNet_results = os.environ.get('nnUNet_results')
```

