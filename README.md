Data-Driven Integration Kernels for the Interpretable Machine Learning of Nonlocal Processes
------------

Savannah L. Ferretti<sup>1</sup>, Jerry Lin<sup>2</sup>, Sara Shamekh<sup>3</sup>, Jane W. Baldwin<sup>1,4</sup>, Michael S. Pritchard<sup>5</sup>, Tom Beucler<sup>6,7</sup>

<sup>1</sup>Department of Earth System Science, University of California Irvine, Irvine, CA, USA  
<sup>2</sup>Department of Computing and Data Science, Boston University, Boston, MA, USA  
<sup>3</sup>The Center for Atmosphere Ocean Science, New York University, New York, NY, USA  
<sup>4</sup>Lamont-Doherty Earth Observatory, Palisades, NY, USA  
<sup>5</sup>NVIDIA Corporation, Santa Clara, CA, USA 
<sup>6</sup>Faculty of Geosciences and Environment, University of Lausanne, Lausanne, VD, CH 
<sup>7</sup>Expertise Center for Climate Extremes, University of Lausanne, Lausanne, VD, CH 

**Status:** Thie paper was submitted to the [15th International Conference on Climate Informatics](https://wp.unil.ch/ci26/). Accepted submissions are automatically forwarded for peer-review at *Environmental Data Science*. We welcome any comments, questions, or suggestions. Please email your feedback to Savannah Ferretti (savannah.ferretti@uci.edu).

**Abstract**: Machine learning models can represent climate processes that are nonlocal in horizontal space, height, and time, often by combining information across these dimensions in highly nonlinear ways. While this can improve predictive skill, it makes learned relationships difficult to interpret and prone to overfitting as the extent of nonlocal information grows. We address this challenge by introducing data-driven integration kernels, a framework that adds structure to nonlocal operator learning by explicitly separating nonlocal information aggregation from local nonlinear prediction. Each spatiotemporal predictor field is first integrated using learnable kernels (defined as continuous weighting functions over horizontal space, height, and/or time), after which a local nonlinear mapping is applied only to the resulting kernel-integrated features and any optional local inputs. This design confines nonlinear interactions to a small set of integrated features and makes each kernel directly interpretable as a weighting pattern that reveals which horizontal locations, vertical levels, and past timesteps contribute most to the prediction. We demonstrate the framework for South Asian monsoon precipitation using a hierarchy of neural network models with increasing structure, including baseline, nonparametric kernel, and parametric kernel models. Across this hierarchy, kernel-based models achieve near-baseline performance with far fewer trainable parameters, showing that much of the relevant nonlocal information can be captured through a small set of interpretable integrations when appropriate structural constraints are imposed.

Project Organization
------------
```
├── LICENSE.md  <- License for code
│
├── README.md   <- Top-level information on this code base/manuscript
│
├── data/
│   ├── raw/          <- Original ERA5 and IMERG V06 data
│   ├── interim/      <- Intermediate data that has been transformed
│   ├── splits/       <- Training, validation, and test sets
│   ├── predictions/  <- Model predictions
│   └── weights/      <- Learned kernel weights
│
├── figs/       <- Generated figures/graphics
│
├── models/     <- Saved baseline NNs, nonparametric kernel NNs, and parametric kernel NNs
│
├── notebooks/  <- Jupyter notebooks for data analysis and visualizations
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
│   │   ├── architectures.py  <- NN architectures
│   │   ├── kernels.py        <- Kernel layers
│   │   ├── train.py          <- Execution script for training models
│   │   └── evaluate.py       <- Execution script for model evaluation
│   │
│   └── utils.py     <- Configuration and utility functions
│
└── environment.yml  <- File for reproducing the analysis environment
```

Acknowledgements
-------
The analysis for this work was performed on NERSC’s [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/). This research was supported by [LEAP NSF-STC](https://leap.columbia.edu/); the US DOE, including the [ASCR](https://www.energy.gov/science/ascr/advanced-scientific-computing-research) Program, the [BER](https://www.energy.gov/science/ber/biological-and-environmental-research) Program Office, and the [PCMDI](https://eesm.science.energy.gov/projects/pcmdi-earth-system-model-evaluation-project) Earth System Model Evaluation Project; [NVIDIA](https://www.nvidia.com/en-us/); NASA’s [NIP-ES](https://science.nasa.gov/earth-science/early-career-opportunities/#h-early-career-investigator-program-in-earth-science); and the Horizon Europe [AI4PEX Project](https://ai4pex.org/) through [SERI](https://www.sbfi.admin.ch/en). Additionally, we thank Fiaz Ahmed and Eric Wengrowski for their input during the early stages of this work, and Jared Sexton for helpful comments on the manuscript prior to submission.

--------
<p><small>This template is based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
