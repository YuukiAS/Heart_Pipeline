# Separate Modality

## 20207 Scout Image (Aortic Structure)
- [x] What is the exact difference between 20207 and 20210? Why 
they don't use 20210?
    
    20207 is multi-slice with single time frame while 20210 is a single slice with multiple timeframe.
- [ ] Incorporate existing repo: https://github.com/cams2b/ukb_aorta/

## 20208 Long Axis
- [ ] Get Mitral valve and Tricuspid valve diameter: *Cardiovascular magnetic resonance reference values of mitral and tricuspid annular dimensions* 
- [ ] Extract landmarks on 3-chamber view: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9708771/pdf/10554_2022_Article_2724.pdf

## 20209 Short Axis

## 20210 Aortic Distensibility Cine
- [ ] Extract luminal parameter

## 20211 Tagged MRI
- [ ] Incorporate existing repo: https://github.com/EdwardFerdian/mri-tagging-strain

## 20212 LVOT Cine
- [ ] Draw some annotations and train the model
- [ ] Allow LVOT diameters to be extracted
- [ ] Determine whether more features can be extracted

## 20213 Phase Contrast
- [ ] Aortic flow: velocity, volume (VTI), area..
- [ ] Number of cusp

## 20214 T1 Mapping
- [ ] Correct some annotations that exceed the myocardium at the bottom

# Other Modality
- [ ] Check Data fields such as PWV https://biobank.ndph.ox.ac.uk/ukb/label.cgi?id=100007
- [ ] ICD coding such as https://biobank.ndph.ox.ac.uk/ukb/label.cgi?id=1712 (Check https://openheart.bmj.com/content/openhrt/9/2/e002039.full.pdf for details)

# All Modality

- [ ] Add some visualization for quality check
- [ ] Add scripts for segmentation based on nnU-Net or U-Mamba

