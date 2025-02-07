---
title: Cardiovascular Magnetic Resonance Imaging (CMR)
---

## Heart Anatomy

The heart is considered to have three segments: the atria, ventricles and the great vessels.

The atria and ventricles are each partitioned into two counterparts, the left and right sides, with two atrioventricular valves (tricuspid valve and mitral valve) that connect the respective ventricle to the atria and the semilunar valves (pulmonary valve, aortic valve) that connect the great vessels to the ventricles.

On the other hand, the right atrium (green chamber in Fig. 1c)  (see Fig. 1b). The right ventricle (blue chamber)  (brown artery in Fig. 1c).

+ The *left atrium (LA)* receives oxygenated blood via the pulmonary veins and is connected to the left ventricle through the *mitral valve* which has two leaflets

+ The *left ventricle (LV)* is the main pumping chamber that pumps blood through the aorta causing the three leaflets *aortic valve* (see Fig. 1b) to open during systole and to close during diastole.

+ The *right atrium (RA)* receives deoxygenated blood from the superior and inferior vena cava and is connected to the right ventricle through the *tricuspid valve* that has three leaflets.

+ The *right ventricle (RV)* pumps blood through the three-leaflets *pulmonary valve* towards the pulmonary artery.

![Heart Anatomy Static](https://raw.githubusercontent.com/YuukiAS/ImageHost/main/heart/20241223115534112.png)
*The blue part corresponds to deoxygenated blood, while the red part corresponds to oxygenated blood*

![Heart Anatomy Dynamic](https://upload.wikimedia.org/wikipedia/commons/7/77/Blood_Circulation.gif?20130928190520)


## Imaging Planes of the Body

The body is divided into three planes:

1. The axial plane (transverse, Z) plane that cuts the body from top to bottom (or superior to inferior).
2. The coronal plane (Y) plane that cuts the body from front to back (or anterior to posterior).
3. The sagittal plane (X) plane that cuts the body from right to left.

![Imaging Plane](https://raw.githubusercontent.com/YuukiAS/ImageHost/main/heart/20241223120003134.png)


## Overview of CMR Protocol for UK Biobank

![UKB Protocol](https://raw.githubusercontent.com/YuukiAS/ImageHost/main/heart/20241223120308544.png)
*(Petersen et al., 2015)*

The information for each modality will be detailed in each section. 

## Reference Ranges of Features

For each feature, the corresponding reference range can be found in `/tasks/reference_range.json` file.

Most features can be indexed using BSA, calculated using the DuBois & DuBois formula: $BSA=0.007184\cdot H^{0.725}\cdot W^{0.425}$, where $H$ is the height in cm and $W$ is the weight in kg (Du Bois & Du Bois, 1916).

TODO: 

- [ ] Number of cusps of aortic valve
- [ ] 3Ch View
- [ ] Enhancement of registration for strain
- [ ] Changes in angle caused by shear (ERC, ECL)
- [ ] More strain calculated from tagged MRI


## References

M.D, R. W. B., Biederman, R. W. W., Doyle, M., & Yamrozik, J. (2008). The Cardiovascular MRI Tutorial: Lectures and Learning. Lippincott Williams & Wilkins.

Viola, F., Del Corso, G., De Paulis, R., & Verzicco, R. (2023). GPU accelerated digital twins of the human heart open new routes for cardiovascular research. Scientific Reports, 13(1), 8230. https://doi.org/10.1038/s41598-023-34098-8

Petersen, S. E., Matthews, P. M., Francis, J. M., Robson, M. D., Zemrak, F., Boubertakh, R., Young, A. A., Hudson, S., Weale, P., Garratt, S., Collins, R., Piechnik, S., & Neubauer, S. (2015). UK Biobank’s cardiovascular magnetic resonance protocol. Journal of Cardiovascular Magnetic Resonance, 18(1), 8. https://doi.org/10.1186/s12968-016-0227-4

Du Bois D, Du Bois EF. A formula to estimate the approximate surface area if height and weight be known. 1916. Nutrition 1989;5:303311; discussion 312–303.
