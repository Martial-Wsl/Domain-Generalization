# Advanced Machine Learning Project - Domain-Generalization

# Dataset
1 - Download the dataset PACS from this link: https://github.com/MachineLearning2020/Homework3-PACS
2 - Place the dataset PACS in the folder:
```
/DescribingTexture/data_api/data/
```

# Pre-trained models
In order to reproduce the values reported in the table, you have to download the pretrained models from this link: https://drive.google.com/drive/folders/17tWDDDPY9fRLrnL3YbwkHrilq12oii2M?usp=sharing

Then, you have to put the "outputs" folder into:

```
/DomainToText_AMLProject/
```

# Fine-tuning
This model is mostly based on the ECCV 2020 paper "Describing Textures using Natural Language".  
More contents are shared on [the project webpage](https://people.cs.umass.edu/~chenyun/texture/).

Code for training and evaluating baseline model are under:

```
/DescribingTexture/models/triplet_match
```

## Environment

To run the code you have to install all the required libraries listed in the "requirements.txt" file.

For example, if you read:

```
torch==1.10.1
```

you have to execute the command:

```
pip install torch==1.10.1
```
