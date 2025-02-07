---
title: Evaluation of Phase Contrast MRI
---


## Description of Modality
A phase contrast sequence is planned on both sagittal and coronal LVOT cines to capture aortic flow and the number of valve cusps. The plane is located at or immediately above the sino-tubular junction at end diastole. The standard velocity encoding (VENC) is 2 m/s but is adjusted upwards based on presence/degree of turbulence seen on the LVOT cines and if time allows. (Petersen et al., 2015)

![Planning of phase contrast MRI](https://raw.githubusercontent.com/YuukiAS/ImageHost/main/heart/20250107095825326.png)*(Petersen et al., 2015)*


![Why it is called phase contrast](https://raw.githubusercontent.com/YuukiAS/ImageHost/main/heart/20250107162517216.png)*(Gatehouse et al., 2005)*

Stationary material is mid-grey (zero-value pixels), with increasing velocities in opposite directions shown brighter (positive pixels) and darker (negative pixels), with an accurate linear relationship to velocity, i.e., $v_{pixel}=VENC\times \frac{phase}{180\degree}$. The velocity encoding value $VENC$ should be available as an adjustable sequence parameter. A small $VENC$ represents a highly velocity-sensitive image (Gatehouse et al., 2005).

## Definition of Features


| Feature Name            | Target  | Description                                                                 | Unit        |
| ----------------------- | ------- | --------------------------------------------------------------------------- | ----------- |

## Features: Definition, How to Derive, and Clinical Relevance

![Aortic valve cusps](https://raw.githubusercontent.com/YuukiAS/ImageHost/main/heart/20250106220922365.png)
*(Rashed et al., 2022)*


Both aortic regurgitant volume and regurgitant fraction depend on the position of the image plane, with systematically lower regurgitation values at more distal positions , which has been attributed to the effect of aortic wall compliance, coronary flow, as well as through plane motion of the aortic root (Iwamoto et al., 2014).

The flow volume per cardiac cycle is the sum of the velocities of the pixels within the ROI multiplied by the area at each cine frame. The ROI around the blood vessel in each frame requires re-drawing for motion and vessel compliance during the cycle (Gatehouse et al., 2005).

Flow displacement is used to describe the eccentricity of the outflow jet.


## References

Petersen, S. E., Matthews, P. M., Francis, J. M., Robson, M. D., Zemrak, F., Boubertakh, R., Young, A. A., Hudson, S., Weale, P., Garratt, S., Collins, R., Piechnik, S., & Neubauer, S. (2015). UK Biobank’s cardiovascular magnetic resonance protocol. Journal of Cardiovascular Magnetic Resonance, 18(1), 8. https://doi.org/10.1186/s12968-016-0227-4

Iwamoto, Y., Inage, A., Tomlinson, G., Lee, K. J., Grosse-Wortmann, L., Seed, M., Wan, A., & Yoo, S.-J. (2014). Direct measurement of aortic regurgitation with phase-contrast magnetic resonance is inaccurate: Proposal of an alternative method of quantification. Pediatric Radiology, 44(11), 1358–1369. https://doi.org/10.1007/s00247-014-3017-x

Gatehouse, P. D., Keegan, J., Crowe, L. A., Masood, S., Mohiaddin, R. H., Kreitner, K.-F., & Firmin, D. N. (2005). Applications of phase-contrast flow and velocity imaging in cardiovascular MRI. European Radiology, 15(10), 2172–2184. https://doi.org/10.1007/s00330-005-2829-3

Rashed, E. R., Dembar, A., Riasat, M., & Zaidi, A. N. (2022). Bicuspid Aortic Valves: An Up-to-Date Review on Genetics, Natural History, and Management. Current Cardiology Reports, 24(8), 1021–1030. https://doi.org/10.1007/s11886-022-01716-2