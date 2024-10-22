# This repository contains the analysis of SMLM images 

This repository contains the analysis of SMLM images predicted by by ANNA PALM ([Ouyang et al. 2019](https://www.nature.com/articles/nbt.4106))

It also contains figures that were used for the paper [J.Bai, W. Ouyang et al.]()

The source code in __/src__ folder is forked from AnetLib in [ANNA PALM repository](https://github.com/imodpasteur/ANNA-PALM/tree/master/AnetLib)


# Table of Contents

* ### [Installation](#installation)
* ### [Usage](#usage)
* ### [License](#license)

# Installation

### 1. Create a virtual environment:

```
conda create -n anna-palm python=3.6.8 -y
conda activate anna-palm
```

### 2. Install jupyter notebook and set the virtuel environment on jupyter notebook
```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=anna-palm
```

### 3. If you have an tensorflow compatible GPU (e.g. Nvidia), and you want to use GPU for training/testing: in the conda virtual enviroment, run the following command:

```
conda install cudatoolkit=9.0 cudnn -y
pip install -r requirements.txt
```


# Usage

### File organisationœ

```
SMLM_Analysis/
│
├── README.md         
│# Overview of the project.
├── LICENSE           
│# MIT License of the project.
├── requirements.txt  
│# Project requirements and dependencies.
│   
├── src/            
│# Directory for project source code.
│   ├── AnetLib   
│   │# Directory for ANNA PALM source code.
│   ├── figure_XXX.py         
│   │# Source code for figures
│   ├── figure_sup_XXX.py         
│   │# Source code for supplementary figures
│
├── code/             # Directory for project source code.
│   ├── main.py       # Main code file.
│   ├── utils.py      # Utility functions.
│   └── ...
│
└── data/             # Directory for project data files.
    ├── dataset.csv   # Sample dataset.
    └── ...
```
### The /script folder contains the notebooks that generate images that used for the figure in the paper

### Analysis of models' performance


## License






