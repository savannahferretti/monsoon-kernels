Data-Driven Integration Kernels for the Interpretable Machine Learning of Nonlocal Processes
------------

[![Preprint](https://img.shields.io/badge/Preprint-arXiv%3A2603.10305-1f6feb?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.10305)

This repository contains the code used to reproduce the analyses in Ferretti et al. (under review in *Environmental Data Science*). The final version will be archived on Zenodo upon acceptance. For questions or feedback, contact Savannah Ferretti (savannah.ferretti@uci.edu).

**Authors & Affiliations:**  
Savannah Ferretti<sup>1</sup>, Jerry Lin<sup>2</sup>, Sara Shamekh<sup>3</sup>, Jane Baldwin<sup>1,4</sup>, Mike Pritchard<sup>5</sup>, & Tom Beucler<sup>6</sup>  
<sup>1</sup>University of California Irvine, Irvine, CA, United States  
<sup>2</sup>Boston University, Boston, MA, United States  
<sup>3</sup>New York University, New York, NY, United States  
<sup>4</sup>Lamont-Doherty Earth Observatory, Palisades, NY, United States  
<sup>5</sup>NVIDIA Corporation, Santa Clara, CA, United States  
<sup>6</sup>University of Lausanne, Lausanne, VD, Switzerland 

**Abstract**: Machine learning models can represent climate processes that are nonlocal in horizontal space, height, and time, often by combining information across these dimensions in highly nonlinear ways. While this can improve predictive skill, it makes learned relationships difficult to interpret and prone to overfitting as the extent of nonlocal information grows. We address this challenge by introducing data-driven integration kernels, a framework that adds structure to nonlocal operator learning by explicitly separating nonlocal information aggregation from local nonlinear prediction. Each spatiotemporal predictor field is first integrated using learnable kernels (defined as continuous weighting functions over horizontal space, height, and/or time), after which a local nonlinear mapping is applied only to the resulting kernel-integrated features and optional local inputs. This design confines nonlinear interactions to a small set of integrated features and makes each kernel directly interpretable as a weighting pattern that reveals which horizontal locations, vertical levels, and past timesteps contribute most to the prediction. We demonstrate the framework for South Asian monsoon precipitation using a hierarchy of neural network models with increasing structure, including baseline, nonparametric kernel, and parametric kernel models. Across this hierarchy, kernel models achieve near-baseline performance with far fewer trainable parameters, indicating that much of the relevant nonlocal information can be captured through a small set of interpretable integrations when appropriate structural constraints are imposed.

How to Use this Repository
------------

All commands should be run from the repository root after cloning:
```
git clone https://github.com/savannahferretti/monsoon-kernels.git
cd monsoon-kernels
```
Create the Conda environment and activate it:
```
conda env create -f environment.yml
conda activate monsoon-kernels
```

Before running any code, edit `scripts/configs.json` to set valid local paths for filepaths (at minimum, `raw`, `interim`, `splits`, `predictions`, `weights`, and `models`). All outputs will be written to these locations. After updating `filepaths`, users can optionally customize other configuration settings, including the spatial/temporal domain in `domain`, the year ranges in `splits`, the chosen predictors and target in `variables`, training hyperparameters in `training`, and the set of model variants in `models`. Model names passed via `--models` must match the name fields in the configuration.

Now, download the ERA5 and IMERG V06 datasets into `data/raw/`, compute derived predictor and target variables stored in `data/interim/`, and generate the training, validation, and test splits written to `data/splits/`:
```
python -m scripts.data.download
python -m scripts.data.calculate
python -m scripts.data.split
```
Train and evaluate all models specified in the configuration file:
```
python -m scripts.models.train
python -m scripts.models.evaluate --split test # or --split valid, for tuning
```
To train and evaluate a single model:
```
python -m scripts.models.train --models baseline_local
python -m scripts.models.evaluate --models baseline_local --split test
```
Or train and evaluate multiple models at once:
```
python -m scripts.models.train --models baseline_local,baseline_nonlocal
python -m scripts.models.evaluate --models baseline_local,baseline_nonlocal --split test
```
To visualize your results, run the Jupyter Notebooks in `notebooks/` to replicate the figures used in the manuscript.

Project Organization
------------
```
├── LICENSE.md        <- License for code
├── README.md         <- Top-level information on this code base/manuscript
├── data/
│   ├── raw/          <- Original ERA5 and IMERG V06 data
│   ├── interim/      <- Intermediate processed data
│   ├── splits/       <- Training, validation, and test sets
│   ├── predictions/  <- Model predictions
│   └── weights/      <- Learned kernel weights
├── figs/             <- Manuscript figures
├── models/           <- Saved model checkpoints for all NNs
├── notebooks/        <- Jupyter notebooks for data analysis and visualization
├── scripts/
│   ├── data/         <- Data processing scripts
│   ├── models/       <- Model building, training, and inferencing scripts
│   └── utils.py      <- Configuration and utility functions
└── environment.yml   <- File for reproducing the analysis environment
```

Acknowledgements
---------
The analysis for this work was performed on NERSC’s [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/). This research was supported by [LEAP NSF-STC](https://leap.columbia.edu/); the US DOE, including the [ASCR](https://www.energy.gov/science/ascr/advanced-scientific-computing-research) Program, the [BER](https://www.energy.gov/science/ber/biological-and-environmental-research) Program Office, and the [PCMDI](https://eesm.science.energy.gov/projects/pcmdi-earth-system-model-evaluation-project) Earth System Model Evaluation Project; [NVIDIA](https://www.nvidia.com/en-us/); NASA’s [NIP-ES](https://science.nasa.gov/earth-science/early-career-opportunities/#h-early-career-investigator-program-in-earth-science); and the Horizon Europe [AI4PEX Project](https://ai4pex.org/) through [SERI](https://www.sbfi.admin.ch/en). Additionally, we thank Fiaz Ahmed and Eric Wengrowski for their contributions during the early stages of this work, Jared Sexton for his helpful comments on the manuscript prior to submission, and Jo Lécuyer for developing an early version of the data-driven integration kernels during a University of Lausanne Master’s internship.
