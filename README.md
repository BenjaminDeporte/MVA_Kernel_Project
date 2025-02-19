# MVA_Kernel_Project
MVA 2024-2025 Kernel Project

Environnement : 
---------------
conda env create -f environment.yml
conda activate kernel_project

Jupyter Kernel (ah ah):
-----------------------
python -m ipykernel install --user --name=kernel_project
jupyter kernelspec list

Repo structure according to DLiP best practice:
-----------------------------------------------

├── LICENSE            <- Information about the license of your code. Companies may have guidelines on how to license code. [See more here](https://choosealicense.com/)
├── README.md          <-  A README file for developers to understand the project setup and instructions.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Graphics and figures for use in reports.
│ 
├── requirements.txt   <- Required libraries and dependencies. 
│
├── pyproject.toml     <- Make the project pip installable with `pip install -e`.
├── src/dlip           <- Source code of the project. `dlip` is the name of the package, you will import it using `import dlip`
│   ├── __init__.py    <- Initializes the 'dlip' Python package.
│   │
│   ├── conf           <- Configuration files for experiments (YAML files managed by Hydra).
│   │   └── train_linear.yaml
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models, to use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
