### [[Drive with Models and Raw Results](https://drive.google.com/drive/folders/1D8AcjTFxKo1hXOC-Ca_y3GaniPVK3yda?usp=drive_link)]
### [[Detailed Project Report](./documentation/Improving%20Visual%20Object%20Tracking%20using%20Deep%20Reinforcement%20Learning.pdf)]

This codebase was created as a submission for CSCE-642 at Texas A&M University by Dineth Gunawardena and Wahib Kapdi.

##### DISLCLAIMER : This code base is based off of Kim Minji's Github repository [Here](https://github.com/byminji/SLTtrack) for SLTTracking. This repository provide a perfect framework to further improve the performance for TransT trackers by adding different RL techniques. We have created our implementation of A2C based SLT in this repository after stripping the repository down to only the necessary files.

# Introduction

Most visual object trackers use Frame Level Tracking. But this does not account for moving and intermediary frames with may have obstacles blocking the subject, motion blur or ambient occlusion. So our project aim to treat this problem as a sequential problem and use the frames and bounding boxes generated in the past as an input to our RL network.

# Improving Object Tracking Networks using Deep Reinforcement Learning

The goal of this project is to improve the existing tracking networks using Reinforcement Learning Algorithms. 
In this project we use Transformer Tracking or [TransT](https://github.com/chenxin-dlut/TransT) coupled with REINFORCE learning to improve performance on Visual Object Tracking Datasets. We use the [GOT10K](http://got-10k.aitestunion.com/) dataset for the same. Using the implementation suggested by [Kim, Minji et. al.](https://arxiv.org/pdf/2208.05810).

# Getting Started

## Setup

We tested the codes in the following environments but other versions may also be compatible.
* CUDA 11.3
* Python 3.9
* PyTorch 1.10.1
* Torchvision 0.11.2

```
# Create and activate a conda environment
conda create -y --name slt python=3.9
conda activate slt

# Install PyTorch
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install requirements
pip install -r requirements.txt
sudo apt-get install libturbojpeg
```
Add the your workspace_path and your local [GOT10k](http://got-10k.aitestunion.com/index) dataset path to [local.py](./pytracking/evaluation/local.py) and [local.py](./ltr/admin/local.py)

## Testing SLT
### [[Official Model from the Paper]](https://drive.google.com/drive/folders/12WWrkx8TrF3-ZhT1AXBdfktKQwQ7ATgh?usp=drive_link) [[Our Trained Version]](https://drive.google.com/drive/folders/16Pe4zr3JSJkzi2j7mwQtC_dzg7VqRuxe?usp=drive_link) 
1. Store one of the two models on your local system.
2. Add the path to this file to [slt_transt.py](pytracking/parameter/slt_transt/slt_transt.py)
3. Then run
```
python pytracking/run_tracker.py slt_transt slt_transt --dataset_name got10k_test
```
4. Then run and submit the generated .zip to the [GOT10K Evaluation](http://got-10k.aitestunion.com/submit_instructions). 
```
python pytracking/util_scripts/pack_got10k_results.py slt_transt slt_transt
```

## Training SLT
### [[TransT Model]](https://drive.google.com/drive/folders/1D8AcjTFxKo1hXOC-Ca_y3GaniPVK3yda?usp=drive_link)
1. Store the baseline model above in yout local system.
2. Add the path to this file to as a pretrained model [slt_transt.py](ltr/train_settings/slt_transt/slt_transt.py)
3. Then run
```
python ltr/run_training.py slt_transt slt_transt
```

## Testing A2C SLT
### [[A2C SLT]](https://drive.google.com/drive/folders/12wqPyJGSx0gszxiyVaCky49bM036ZF-b?usp=drive_link)
1. Store this model on your local system.
2. Add the path to this file to [ac_slt_transt.py](pytracking/parameter/ac_slt_transt/ac_slt_transt.py)
3. Then run
```
python pytracking/run_tracker.py ac_slt_transt ac_slt_transt --dataset_name got10k_test
```
4. Then run and submit the generated .zip to the [GOT10K Evaluation](http://got-10k.aitestunion.com/submit_instructions). 
```
python pytracking/util_scripts/pack_got10k_results.py ac_slt_transt ac_slt_transt
```
## Training A2C SLT
### [[TransT Model]](https://drive.google.com/drive/folders/1D8AcjTFxKo1hXOC-Ca_y3GaniPVK3yda?usp=drive_link)
1. Store the baseline model above in yout local system.
2. Add the path to this file to as a pretrained model [ac_slt_transt.py](ltr/train_settings/ac_slt_transt/ac_slt_transt.py)
3. Then run
```
python ltr/run_training.py ac_slt_transt ac_slt_transt
```

## Results
![image](https://github.com/user-attachments/assets/00205559-66f0-44b6-8cf5-e8f8fc1348a1)

In the results, it is evident that SLT improves on the Baseline. 
In case of our A2C SLT, we failed to train it for long enough, but it still surpasses the Baseline on all measures and reaches pretty close to the SLT Tracker.


## Acknowledgments
SLTTracking was not developed by us, it is taken from the 2022 ECCV paper by Kim, Minji et al.
```bibtex
@inproceedings{SLTtrack,
  title={Towards Sequence-Level Training for Visual Tracking},
  author={Kim, Minji and Lee, Seungkwan and Ok, Jungseul and Han, Bohyung and Cho, Minsu},
  booktitle={ECCV},
  year={2022}
}
```
We have used it's code base here as our starting point. Here is link to it's github [Github](https://github.com/byminji/SLTtrack/tree/master).
SLTtrack is developed upon [PyTracking](https://github.com/visionml/pytracking) library,
also borrowing from [TransT](https://github.com/chenxin-dlut/TransT).
We would like to thank the authors for providing great frameworks and toolkits.


## Contact
Dineth Gunawardena: pgunawardena@tamu.edu \
Wahib Kapdi: wahibkapdi@tamu.edu
