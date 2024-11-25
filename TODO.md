# Separate Modality

## 20207 Scout Image (Aortic Structure)
- [x] What is the exact difference between 20207 and 20210? Why 
they don't use 20210?
    
    20207 is multi-slice with single time frame while 20210 is a single slice with multiple timeframe.
- [x] Incorporate existing repo: https://github.com/cams2b/ukb_aorta/
- [ ] Add units for the extracted features

## 20208 Long Axis
- [x] Get Mitral valve and Tricuspid valve diameter: *Cardiovascular magnetic resonance reference values of mitral and tricuspid annular dimensions* 
- [ ] Extract landmarks on 3-chamber view: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9708771/pdf/10554_2022_Article_2724.pdf  Can I use Atlas to do it?

## 20209 Short Axis
- [ ] Fix the problem that rotation and torsion are not calculated correctly, add visualization

## 20210 Aortic Distensibility Cine
- [x] Extract luminal parameter

## 20211 Tagged MRI
- [x] Incorporate existing repo: https://github.com/EdwardFerdian/mri-tagging-strain

## 20212 LVOT Cine
- [x] Draw some annotations and train the model
- [x] Allow LVOT diameters to be extracted
- [x] Determine whether more features can be extracted

## 20213 Phase Contrast
- [ ] Add segmentation code
- [ ] Aortic flow: velocity, volume (VTI), area..
- [ ] Number of cusp: Incorporate the code

## 20214 T1 Mapping
- [x] Correct some annotations that exceed the myocardium at the bottom
- [ ] Add segmentation code
- [ ] Check whether results meet the reference range
- [ ] Improve Segmentation Quality


# Others
- [ ] Try out https://microsoft.github.io/BiomedParse/. Can it be used to improve segmentation quality or the annotation tool?

# All Modality

- [ ] Make boxplots for all features, add reference range for each feature
- [ ] Improve bias correction for LVOT and Phase contrast.
- [ ] Possible Enhancement1: Complex time series features using TSFEL at step4
- [ ] Possible Enhancement2: Interact with LLM, giving an input figure

