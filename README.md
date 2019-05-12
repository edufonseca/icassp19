
## Learning Sound Event Classifiers from Web Audio with Noisy Labels

This repository contains the code corresponding to the following <a href="https://arxiv.org/abs/1901.01189" target="_blank">ICASSP 2019 paper</a>. If you use this code or part of it, please cite:

>Eduardo Fonseca, Manoj Plakal, Daniel P. W. Ellis, Frederic Font, Xavier Favory, Xavier Serra, "Learning Sound Event Classifiers from Web Audio with Noisy Labels", In proceedings of ICASSP 2019, Brighton, UK

The framework comprises all the basic stages: feature extraction, training, inference and evaluation. After loading the FSDnoisy18k dataset, log-mel energies are computed and a CNN baseline is trained and evaluated. The code also allows to test four noise-robust loss functions. Please check our paper for more details. The system is implemented in Keras and TensorFlow.

The FSDnoisy18k dataset described in our ICASSP 2019 paper is available through Zenodo from its companion site: <a href="http://www.eduardofonseca.net/FSDnoisy18k/" target="_blank">http://www.eduardofonseca.net/FSDnoisy18k/</a>. 

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

  - `ctrl.train_data: all` # (or any other train subset)
  - `loss.type: CCE` # this is standard cross entropy loss
 
#### (2) Execute the code.

## Baseline system details

Incoming audio is transformed to 96-band, log-mel spectrogram as input representation.
To deal with the variable-length clips, we use time-frequency patches of 2s (which is equivalent to 100 frames of 40ms with 50% overlap). Shorter clips are replicated while longer clips are trimmed in several patches inheriting the clip-level label (this is the meaning of the parameter `ctrl.load_mode = varup` in the `config/*.yaml` file).


The model used is a CNN (3 conv layers + 1 dense layer) following that of <a href="https://arxiv.org/abs/1608.04363" target="_blank">this paper</a>, with two main changes. First, we include Batch Normalization (BN) between each convolutional layer and ReLU non-linearity. Second, we use *pre-activation*, a technique initially devised in <a href="https://arxiv.org/abs/1603.05027" target="_blank">deep residual networks</a> which essentially consists of applying BN and ReLU as pre-activation before each convolutional layer.
It was proved beneficial for acoustic scene classification in <a href="https://arxiv.org/abs/1806.07506" target="_blank">this paper</a>, where it showed convenient generalization properties. Likewise, in preliminary experiments with FSDnoisy18k it was shown to slightly improve the classification accuracy. The baseline system has 531,624 weights and its architecture is summarized in the next figure.

<p align="center">

<img src="/figs/baseline_system_archi_v1.png" alt="baseline" width="550"/>

</p>

As for the learning strategy, the default loss function is categorical cross-entropy (CCE), the batch size is 64, and we use Adam optimizer with initial learning rate of 0.001, which is halved whenever the validation accuracy plateaus for 5 epochs. The training samples are shuffled between epochs. Earlystopping is adopted with a patience of 15 epochs on the validation accuracy. To this end, a 15% validation set is split randomly from the training data of every class. This validation split is the random 15% of every class, considering both *clean* and *noisy* subsets together. Preliminary experiments revealed that this provides slightly better results if compared to using **only** the clean subset for validation (which amounts to roughly 10% of the training set, but it is highly imbalanced class-wise, from 6.1% to 22.4%). 

On inference, the prediction for every clip is obtained by computing predictions at the patch level, and aggregating them with geometric mean to produce a clip-level prediction.

The goal of the baseline is to give a sense of the classification accuracy that a well-known architecture can attain and not to maximize the performance. 
Extensive hyper-parameter tuning or additional model exploration was not conducted.

 
## Contact

You are welcome to contact me privately should you have any question/suggestion or if you have any problems running the code at eduardo.fonseca@upf.edu. You can also create an issue.


