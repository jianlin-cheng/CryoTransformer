# CryoTransformer: CryoTransformer: Leveraging Transformer Based Approach for Identifying and Extracting Protein Particles from Cryo-EM Micrographs 

CryoTransformer is a powerful and accurate particle-picking framework using Residual Network (ResNet) and Transformer. CryoTransformer was trained using the largest diverse labelled CryoPPP dataset for the first time. It recognizes and extracts abundance amount of true protein particles from the input micrographs while maintaining low false-positive rates. We performed rigorous evaluation, comparing our method with existing AI based methods and showcasing its robustness through multiple labels of evaluation. Our model outperformed the current state-of-the art methods and is poised to greatly facilitate the automation of Cryo-EM particle recognition.

---
## Overview of CryoTransformer Pipeliine

![Alt text](<visuals/CryorTransformer_pipeline.jpg>)

## Installation

#### Clone project
```
git clone https://github.com/jianlin-cheng/CryoTransformer.git
cd CryoTransformer/
```
#### Download trained models
```
curl https://calla.rnet.missouri.edu/CryoTransformer/pretrained_model.tar.gz --output pretrained_model.tar.gz
tar -xvf pretrained_model.tar.gz
rm pretrained_model.tar.gz
```
#### Download training data (if required)
```
curl https://calla.rnet.missouri.edu/CryoTransformer/train_val_test_data.tar.gz --output train_val_test_data.tar.gz
tar -xvf train_val_test_data.tar.gz
rm train_val_test_data.tar.gz
```
#### Download test data
```
curl https://calla.rnet.missouri.edu/CryoTransformer/test_data.tar.gz --output test_data.tar.gz
tar -xvf test_data.tar.gz
rm test_data.tar.gz
```
#### Create conda environment
```
conda env create -f environment.yml
conda activate CryoTransformer
```



## Training and Reproducing Results (if required)

```
python train.py
```
```
Optional Arguments:
TODO

Example Usage:
    python train.py --TODO
```
## Prediction

#### Prediction on Test Data 
This code generates the predicted proteins encircled in Micrographs along with the .box and .star files. 
```
python predict.py TODO
```
```
Optional Arguments:
TODo
```

## Training Data Statistics

Data statistics for Training, validating, and Testing CryoTransformer (* Theoretical weight)


| SN | EMPIAR ID | Type of Protein         | Micrograph Size   | Total Structure Weight (kDa) | Training Micrographs | Validation Micrographs | Test Micrographs | Total  Micrographs |
| -- | --------- | ----------------------- | ------------ | ---------------------------- | -------------------- | ---------------------- | ---------------- | ----------------- |
| 1  | 11183​    | Signaling Protein       | (5760, 4092) | 139.36                       | 250                  | 25                     | 25               | 300               |
| 2  | 11057​    | Hydrolase               | (5760, 4092) | 149.43                       | 250                  | 25                     | 20               | 295               |
| 3  | 11051​    | Transcription/DNA/RNA   | (3838, 3710) | 357.31                       | 250                  | 25                     | 25               | 300               |
| 4  | 10852​    | Signaling Protein       | (5760, 4092) | 157.81                       | 270                  | 40                     | 33               | 343               |
| 5  | 10816​    | Transport Protein       | (7676, 7420) | 166.62                       | 250                  | 25                     | 25               | 300               |
| 6  | 10760​    | Membrane Protein        | (3838, 3710) | 321.69                       | 250                  | 25                     | 25               | 300               |
| 7  | 10737​    | Membrane Protein        | (5760, 4092) | 155.83                       | 250                  | 25                     | 17               | 292               |
| 8  | 10671​    | Signaling Protein       | (5760, 4092) | 77.14                        | 250                  | 25                     | 23               | 298               |
| 9  | 10590​    | Transport Protein       | (3710, 3838) | 1000\*                       | 250                  | 25                     | 21               | 296               |
| 10 | 10526​    | Ribosome (50S)          | (7676, 7420) | 1085.81                      | 180                  | 20                     | 20               | 220               |
| 11 | 10444​    | Membrane Protein        | (5760, 4092) | 295.89                       | 250                  | 25                     | 21               | 296               |
| 12 | 10406​    | Ribosome (70S)          | (3838, 3710) | 632.89                       | 200                  | 20                     | 19               | 239               |
| 13 | 10387​    | Viral Protein           | (3710, 3838) | 185.87                       | 250                  | 25                     | 24               | 299               |
| 14 | 10291​    | Transport Protein       | (3710, 3838) | 361.39                       | 250                  | 25                     | 25               | 300               |
| 15 | 10289​    | Transport Protein       | (3710, 3838) | 361.39                       | 250                  | 25                     | 25               | 300               |
| 16 | 10240​    | Lipid Transport Protein | (3838, 3710) | 171.72                       | 250                  | 25                     | 24               | 299               |
| 17 | 10184​    | Aldolase                | (3838, 3710) | 150\*                        | 250                  | 25                     | 21               | 296               |
| 18 | 10096​    | Viral Protein           | (3838, 3710) | 150\*                        | 250                  | 25                     | 25               | 300               |
| 19 | 10077​    | Ribosome (70S)          | (4096, 4096) | 2198.78                      | 250                  | 25                     | 25               | 300               |
| 20 | 10075​    | Bacteriophage MS2       | (4096, 4096) | 1000\*                       | 250                  | 25                     | 24               | 299               |
| 21 | 10059​    | Transport Protein       | (3838, 3710) | 317.88                       | 250                  | 25                     | 16               | 291               |
| 22 | 10005​    | Transport Protein       | (3710, 3710) | 272.97                       | 22                   | 4                      | 3                | 29                |
|    |           | Total Micrographs       |              |                              | 5,172                | 534                    | 486              | 6,192             |



## Independent Test Data Statistics
Data statistics used for independent Testing sourced from EMPIAR repository

| SN | EMPIAR ID         | Type of Protein   | Micrograph Size   | Total Structure Weight (kDa) | # of Micrographs |
| -- | ----------------- | ----------------- | ------------ | ---------------------------- | --------------------- |
| 1  | 10081             | Transport Protein | (3710, 3838) | 298.57                       | 997                   |
| 2  | 10532             | Viral Protein     | (4096, 4096) | 191.76                       | 1,556                 |
| 3  | 10093             | Membrane Protein  | (3838, 3710) | 779.4                        | 1,873                 |
| 4  | 10345             | Signaling Protein | (3838, 3710) | 244.68                       | 1,644                 |
|    | Total Micrographs |                   |              |                              | 6,070                 |


Data statistics used for independent Testing sourced from CryoPPP dataset (* Theoretical weight)

| SN | EMPIAR ID | Type of Protein   | Micrograph Size | Total Structure Weight (kDa) | # of Micrographs |
| -- | --------- | ----------------- | --------------- | ---------------------------- | --------------------- |
| 1  | 10017     | β -galactosidase  | (4096, 4096)    | 450\*                        | 84                    |
| 2  | 10081     | Transport Protein | (3710, 3838)    | 298.57                       | 300                   |
| 3  | 10093     | Membrane Protein  | (3838, 3710)    | 779.4                        | 295                   |
| 4  | 10345     | Signaling Protein | (3838, 3710)    | 244.68                       | 295                   |
| 5  | 10532     | Viral Protein     | (4096, 4096)    | 191.76                       | 300                   |
| 6  | 11056     | Transport Protein | (5760, 4092)    | 88.94                        | 305                   |
|    |           | Total Micrographs |                 |                              | 1,579                 |


## Rights and Permissions
Open Access \
This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.



## Cite this work
If you use the code or data associated with this research work or otherwise find this data useful, please cite:

## CryoTransformer
TO Update


## Dataset
If you use the code or data associated with this research work or otherwise find this data useful, please cite: \
@article {Dhakal2023, \
	author = {Dhakal, Ashwin and Gyawali, Rajan and Wang, Liguo and Cheng, Jianlin}, \
	title = {A large expert-curated cryo-EM image dataset for machine learning protein particle picking}, \
	year = {2023}, \
    volume = {10}, \
    issue = {1}, \
	doi = {10.1038/s41597-023-02280-2}, \
	journal = {Scientific Data}, \
    url = { https://doi.org/10.1038/s41597-023-02280-2 }
}