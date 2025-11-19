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
│   ├── nn/            <- Saved NN models
│   └── sr/            <- Saved PySR models
|
├── notebooks/         <- Jupyter notebooks for data analysis and visualizations
│
├── scripts/           
│   ├── data/          <- Scripts for downloading, and calculating input/target terms, and splitting data
│   ├── nn/            <- Scripts for training/evaluating baseline NNs, non-parametric kernel NNs, and parametric kernel NNs
│   └── sr/            <- Scripts for running PySR and summarizing discovered equations 
│
└── environment.yml    <- File for reproducing the analysis environment
```

Acknowledgements
-------
The analysis for this work has been performed on NERSC's [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/). This research was funded by ```<GRANTOR>```.```<NAME>``` provided helpful feedback on the first draft. Thanks to our colleagues at ```<ORGANIZATION>``` for their continued support.

--------
<p><small>This template is based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
