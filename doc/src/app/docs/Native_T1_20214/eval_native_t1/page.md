---
title: 
---


## Description of Modality

The Shortened Modified Look-Locker Inversion recovery technique (ShMOLLI, WIP780B) is implemented on the scanner in order to perform native (non-contrast) myocardial T1 mapping. Native T1 mapping in one midventricular short axis is added to allow myocardial tissue characterisation without the use of contrast agents. (Petersen et al., 2015)

## Definition of Features

| Feature Name      | Target    | Description                                  | Unit |
| ----------------- | --------- | -------------------------------------------- | ---- |
| Myocardium-Global | Native T1 | T1 value for the LV mid-wall myocardium      | ms   |
| Myocardium-IVS    | Native T1 | T1 value for the LV intra-ventricular septum | ms   |
| Myocardium-FW     | Native T1 | T1 value for the LV free wall                | ms   |
| LV Blood Pool     | Native T1 | T1 value for the LV blood pool               | ms   |
| RV Blood Pool     | Native T1 | T1 value for the RV blood pool               | ms   |


## Features: Definition, How to Derive, and Clinical Relevance


```
evaluate_t1_uncorrected()
```

### Corrected T1 Values

The T1 value of blood is higher than the T1 value of the myocardium, thus the T1 value of the myocardium is expected to be influenced by both the T1 value and the amount of the blood in the myocardium, since the normal myocardial blood volume (MBV) averages approximately 6% of the myocardium (McCommis et al., 2007). Therefore, the T1 values of blood have been shown to vary due to a nmumber of factors such as sex and age, leading to the influence of the interpretation of the myocardial T1 values and the obscuring of the true T1 value of the myocardium (Nickander et al., 2016).

## References

Petersen, S. E., Matthews, P. M., Francis, J. M., Robson, M. D., Zemrak, F., Boubertakh, R., Young, A. A., Hudson, S., Weale, P., Garratt, S., Collins, R., Piechnik, S., & Neubauer, S. (2015). UK Biobank’s cardiovascular magnetic resonance protocol. Journal of Cardiovascular Magnetic Resonance, 18(1), 8. https://doi.org/10.1186/s12968-016-0227-4

McCommis, K., Goldstein, T., Zhang, H., Misselwitz, B., Gropler, R., & Zheng, J. (2007). Quantification of Myocardial Blood Volume During Dipyridamole and Doubtamine Stress: A Perfusion CMR Study. Journal of Cardiovascular Magnetic Resonance, 9(5), 785–792. https://doi.org/10.1080/10976640701545206

Nickander, J., Lundin, M., Abdula, G., Sörensson, P., Rosmini, S., Moon, J. C., Kellman, P., Sigfridsson, A., & Ugander, M. (2016). Blood correction reduces variability and gender differences in native myocardial T1 values at 1.5 T cardiovascular magnetic resonance – a derivation/validation approach. Journal of Cardiovascular Magnetic Resonance, 19(1), 41. https://doi.org/10.1186/s12968-017-0353-7


