Data-Driven Integration Kernels for the Interpretable Machine Learning of Nonlocal Processes
------------

**Status:** This manuscript is currently in preparation and will be submitted to the [15th International Conference on Climate Informatics](https://wp.unil.ch/ci26/). Accepted submisisons are automatically forwarded for peer-review at *Environmental Data Science*. We welcome any comments, questions, or suggestions. Please email your feedback to Savannah Ferretti (savannah.ferretti@uci.edu).

**Abstract**: Machine learning models can represent climate processes that are nonlocal in space, height, and time, but they often combine information across these dimensions in highly nonlinear ways. While this can improve predictive skill, it makes the learned relationships difficult to interpret and prone to overfitting as the nonlocal context grows. We address this by introducing \emph{data-driven integration kernels}, a framework that adds structure to nonlocal operator learning by explicitly separating nonlocal information aggregation from local nonlinear prediction. In this approach, each spatiotemporal predictor field is first integrated using learnable kernels defined as continuous weighting functions over space, height, and/or time, after which a local nonlinear mapping is applied only to the resulting kernel-integrated features and any optional local inputs. This design limits nonlocal interactions to a small set of shared integration patterns and makes each kernel directly interpretable as a weighting pattern that indicates which horizontal distances, vertical levels, and past timesteps contribute most to a prediction. We demonstrate the framework for South Asian monsoon precipitation using a hierarchy of neural network models with increasing structure, including unconstrained baseline models, nonparametric kernel models, and compact parametric kernel models. Across this hierarchy, kernel-based models retain predictive skill comparable to the baseline while using far fewer trainable parameters, showing that much of the relevant nonlocal information can be captured through a small set of physically indexed integrations.

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
│   ├── predictions/   <- Model predictions
│   └── weights/       <- Learned kernel weights
│
├── figs/              <- Generated figures/graphics
│
├── models/
│   ├── baseline/      <- Saved baseline NN models
│   ├── nonparametric/ <- Saved nonparametric kernel NN models
│   └── parametric/    <- Saved parametric kernel NN models
│
├── notebooks/         <- Jupyter notebooks for data analysis and visualizations
│
├── scripts/
│   ├── data/
│   │   ├── classes/      <- Data processing classes
│   │   ├── download.py   <- Execution script for downloading raw data
│   │   ├── calculate.py  <- Execution script for calculating derived variables
│   │   └── split.py      <- Execution script for creating train/valid/test splits
│   │
│   ├── models/
│   │   ├── classes/          <- Model building, training, and inferencing classes
│   │   ├── architectures.py  <- Neural network architectures
│   │   ├── kernels.py        <- Kernel layers
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
