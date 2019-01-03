
## Learning Sound Event Classifiers from Web Audio with Noisy Labels

This repository contains the code corresponding to the following paper. If you use this code or part of it, please cite:

>Eduardo Fonseca, Manoj Plakal, Daniel P. W. Ellis, Frederic Font, Xavier Favory, Xavier Serra, "Learning Sound Event Classifiers from Web Audio with Noisy Labels", Submitted to *Proc. IEEE ICASSP 2019*, Brighton, UK, 2019

The framework comprises all the basic stages: feature extraction, training, inference and evaluation. After loading the FSDnoisy18k dataset, log-mel energies are computed and a CNN baseline is trained and evaluated. The code also allows to test four noise-robust loss functions. Please check our ICASSP2019 paper for more details.

**NOTES**: 
- The code is available and it is functional. 
- The FSDnoisy18k dataset described in the ICASSP2019 paper is available from its companion site: <a href="http://www.eduardofonseca.net/FSDnoisy18k/" target="_blank">http://www.eduardofonseca.net/FSDnoisy18k/</a>. 
- In addition, an extended description of the baseline system will be made available soon.

## Dependencies
This framework is tested on Ubuntu 17.10 using a conda environment. To duplicate the conda environment:

`conda create --name <envname> --file requirements.txt`


## Directories and files

`config/` includes a `*.yaml` file with the parameters for the experiment  
`logs/` folder where to include output files per experiment  

`main.py` is the main script  
`data.py` contains the data generators  
`feat_extract.py` contains feature extraction code  
`architectures.py` contains the architecture for the baseline system  
`utils.py` some basic utilities  
`eval.py` evaluation code  
`losses.py` definition of several loss functions  



## Usage

#### (0) Download the dataset:

Download FSDnoisy18k from Zenodo through the <a href="http://www.eduardofonseca.net/FSDnoisy18k/" target="_blank">dataset companion site</a>, unzip it and locate it in a given directory.

#### (1) Edit `config/*.yaml` file:

The goal is to define the parameters of the experiment. The file is structured with self-descriptive sections. The most important parameters are: 

`ctrl.dataset_path`: path where the dataset is located, eg, `/data/FSDnoisy18k/`.   
`ctrl.train_data`: define the subset of training data to consider in the experiment. To be decided among: `['all', 'noisy', 'noisy_small', 'clean']` (see paper)   
`loss.q_loss`: this is an example of a hyper-parameter of a loss function, according to the paper. For example, `q_loss` corresponds to `q` in equation (3) of the paper and `reed_beta` corresponds to `beta` in equation (2).  
`loss.type`: defines the loss function. To be decided among:

  - `CCE`: categorical_crossentropy aka cross entropy loss
  - `lq_loss`: L_q loss
  - `CCE_max`: CCE loss & discard loss values using maximum-based threshold
  - `CCE_outlier`: CCE loss & discard loss values using outlier-based threshold
  - `bootstrapping`: L_soft loss
  - `lq_loss_origin`: L_q loss applied selectively based on data origin*
  - `CCE_max_origin`: CCE_max applied selectively based on data origin*
  - `CCE_outlier_origin`: CCE_outlier applied selectively based on data origin*
  - `bootstrapping_origin`: L_soft loss applied selectively based on data origin*

*The selective application of the loss functions makes sense when training with the entire train set (that is, considering clean and noisy data), ie `ctrl.train_data: all ` (see paper).

The rest of the parameters should be rather intuitive.


#### (2) Execute the code by:
- activating the conda env 
- run, for instance: `CUDA_VISIBLE_DEVICES=0 KERAS_BACKEND=tensorflow python main.py -p config/params.yaml &> logs/output_file.out`

In the first run, log-mel features are extracted and saved. In the following times, the code detects that there is a feature folder. It *only* checks the folder; not the content. If some feature extraction parameters are changed, the program won’t know it.

#### (3) See results:

You can check the `logs/*.out`. Results are shown in a table (you can search for the string `ACCURACY - MICRO` and it will take you to them).


## Reproducing the baseline

#### (1) Edit `config/*.yaml` file

  - `ctrl.train_data: clean` # (or any other train subset)
  - `loss.type: CCE` # this is standard cross entropy loss
 
#### (2) Execute the code.

## Baseline system details

Add Figure

 
## Contact

You are welcome to contact me privately should you have any question/suggestion or especially if you have any problems running the code at eduardo.fonseca@upf.edu. You can also create an issue.


