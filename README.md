# Overview
This is a modified version of [Wenjia Bai's Pipeline](https://github.com/baiwenjia/ukbb_cardiac) to process UKBiobank data.


## Preparation

Please make sure `data_raw_dir` in `config.py` have the following structure:

```
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
├── 20212
├── 20213
```

where `retest` should be the same as `retest_suffix` field in `config.py`.