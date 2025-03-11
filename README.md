# MVA_Kernel_Project
Challenge from the MVA course Kernel methods for machine learning.

## Description
This challenge is a sequence classification task: predicting whether a DNA sequence region is binding site to a specific transcription factor.

Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes. Genome-wide protein-DNA binding maps can be profiled using some experimental techniques and thus all genomics can be classified into two classes for a TF of interest: bound or unbound. In this challenge, we will work with three datasets corresponding to three different TFs.

## Installation
### Prerequisites
Ensure you have Python 3.x installed. You can check by running:
```bash
python --version
```

### Create a Virtual Environment
#### Using venv
```bash
python -m venv venv
source venv/bin/activate
```

#### Using Conda
```bash
conda env create -f environment.yml
conda activate kernel_project
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
```bash
python start.py
```

## Running Tests
Test algorithms:
```bash
python -m test.tests_algos
```

Test Kernels
```bash
python -m test.tests_kernels
```
