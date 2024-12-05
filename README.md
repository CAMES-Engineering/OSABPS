# Open-Source Automatic Bowel Preparation Score (OSABPS)

Open Source Automatic Bowel Preparation Score: Code, weights and updates for the OSABPS metric for assessing colonoscopy bowel preparation 

Related to publication Development and validation of the Open-Source Automatic Bowel Preparation Scale in Gastrointestinal Endoscopy [https://doi.org/10.1016/j.gie.2024.11.022]

## OSABPS

OSABPS is an open-source AI tool for assessing bowel preparation quality in colonoscopies. Traditional methods like the Boston Bowel Preparation Scale (BBPS) have shown limitations in correlating with polyp detection rates. OSABPS aims to improve this by providing an automated, objective measure that directly correlates with polyp detection.  

### How It Works
OSABPS analyzes colonoscopy frames - single image input - to quantify the amount of fecal matter present. It calculates a "fecal ratio" (approximation of proportion of fecal pixels to mucosal pixels), with 0 being the optimal score indicating perfect bowel cleansing. The model was trained on 50,000 frames from 20 colonoscopies and validated internally on 1,405 colonoscopies from three hospitals (see paper in DOI link above), as well as externally on 5,525 frames from a public dataset (Nerthus, Simula.no).

### Key Features
- Automated Scoring: No human bias, instant assessment of bowel prep quality.
- Open Source: Free to use, modify, and implement. Apache 2
- Correlation with PDR: Significant correlation with polyp detection rate.

### Value of Open Source Models
- Transparency: Source code is available for inspection and verification.
- Customization: Developers can modify the model to fit specific needs. e.g. for feature extraction
- Community Contributions: Collaborative improvement and bug fixes.
- Cost-Effective: No licensing fees for implementation.
- Educational Resource: Valuable for learning and research purposes.

Feel free to use, contribute, or provide feedback. Contact Morten, research.mbs@pm.me

## How to make it work for you


### Example usage
We have made some examples, see the usage_examples folder, and will continue to add more as we use the model ourself.
_Nerthus_: Example for downloading data, and python code to analyse the whole dataset, and plotting examples of AutoEncoder output (image and scalar)


### Versions
Version used in [/10.1016/j.gie.2024.11.022]: OSABPS_MODEL_V1_OCT2023.h5


### Debuggin'

```
OSError: Unable to synchronously open file (file signature not found)
```
I encountered the above error when downloading the model file for another project. Github only provided a 2kb file for the 2.5mb model, both during direct download, but also zipped download of the whole repo. The full file could be downloaded by choosing the "raw" mode under the file.


## Other information
##### LLM usage
ChatGPT 1o preview was used for making scripts and code adhere to PEP8 standards and for cleaning code, i.e. removing dev-comments, removing unused imports and vars, etc
