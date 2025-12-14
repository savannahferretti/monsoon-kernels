Data-Driven Integration Kernels for the Interpretable Machine Learning of Nonlocal Processes
------------

By Savannah L. Ferretti <sup>1</sup>, Tom Beucler <sup>2</sup>, Michael S. Pritchard<sup>3</sup>, & Jane W. Baldwin<sup>1,4</sup>

<sup>1</sup>Department of Earth System Science, University of California, Irvine, Irvine, CA, USA  
<sup>2</sup>Faculty of Geosciences and Environment, University of Lausanne, Lausanne, VD, CH  
<sup>3</sup>NVIDIA Corporation, Sanata Clara, CA, USA  
<sup>4</sup>Lamont-Doherty Earth Observatory, Columbia University, Palisades, NY, USA  

**Status:** This manuscript is currently in preparation and will be submitted to the [15th International Conference on Climate Informatics](https://wp.unil.ch/ci26/). Accepted submisisons are automatically forwarded for peer-review at *Environmental Data Science*. We welcome any comments, questions, or suggestions. Please email your feedback to Savannah Ferretti (savannah.ferretti@uci.edu).

**Key Points**:
- Point 1
- Point 2
- Point 3
  
**Abstract**: Insert abstract text here.

Project Organization
------------
```
├── LICENSE.md         <- License for code
│
├── README.md          <- Top-level information on this code base/manuscript
│
├── data/
│   ├── raw/           <- Original ERA5 and IMERG V06 data
│   ├── interim/       <- Intermediate data that has been transformed
│   ├── splits/        <- Training, validation, and test sets
│   └── results/       <- Model predictions (and skill metrics)
│
├── figs/              <- Generated figures/graphics
│
├── models/
│   ├── baseline/      <- Saved baseline NN models
│   ├── nonparametric/ <- Saved non-parametric kernel NN models
│   └── parametric/    <- Saved parametric kernel NN models
│
├── notebooks/         <- Jupyter notebooks for data analysis and visualizations
│
├── scripts/
│   ├── data/
│   │   ├── classes/   <- Data processing classes (DataDownloader, DataCalculator, DataSplitter, PatchDataLoader, PredictionWriter)
│   │   ├── download.py   <- Execution script for downloading raw data
│   │   ├── calculate.py  <- Execution script for calculating derived variables
│   │   └── split.py      <- Execution script for creating train/valid/test splits
│   │
│   ├── models/
│   │   ├── classes/      <- Model classes (Trainer, Inferencer, ModelFactory)
│   │   ├── architectures.py  <- Neural network architectures (MainNN, BaselineNN, KernelNN)
│   │   ├── kernels.py        <- Kernel layers (NonparametricKernelLayer, ParametricKernelLayer)
│   │   ├── train.py          <- Execution script for training models
│   │   └── evaluate.py       <- Execution script for model evaluation
│   │
│   └── utils.py       <- Configuration and utility functions
│
└── environment.yml    <- File for reproducing the analysis environment
```

Acknowledgements
-------
The analysis for this work has been performed on NERSC's [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/). This research was funded by ```<GRANTOR>```.```<NAME>``` provided helpful feedback on the first draft. Thanks to our colleagues at ```<ORGANIZATION>``` for their continued support.

--------
<p><small>This template is based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
