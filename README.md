### [[Drive with Models and Raw Results](https://drive.google.com/drive/folders/1D8AcjTFxKo1hXOC-Ca_y3GaniPVK3yda?usp=drive_link)]
### [[Detailed Project Report](./documentation/Improving%20Visual%20Object%20Tracking%20using%20Deep%20Reinforcement%20Learning.pdf)]
### [[Video Summary of the work done and results](https://www.youtube.com/watch?v=75LUAj6q2V8)]

This codebase was created as a submission for CSCE-642 at Texas A&M University by Dineth Gunawardena and Wahib Kapdi.

##### DISLCLAIMER : This code base is based off of Kim Minji's Github repository [Here](https://github.com/byminji/SLTtrack) for SLTTracking. This repository provide a perfect framework to further improve the performance for TransT tracker by using different RL techniques. We have created our implementation of A2C based SLT in this repository after stripping the repository down to only the necessary files.

# Introduction

Traditional visual object trackers often rely on frame-level tracking, which struggles with challenges like occlusion, motion blur, or ambient conditions. This project addresses these issues by treating tracking as a sequential decision problem. By incorporating past frames and bounding boxes as inputs to a reinforcement learning (RL) network, we aim to reduce random perturbations and improve tracking robustness for difficult scenarios.

# Improving Object Tracking Networks using Deep Reinforcement Learning

This project seeks to enhance visual object tracking performance by integrating RL algorithms with existing tracking frameworks. Specifically, we use [Transformer Tracking (TransT)](https://github.com/chenxin-dlut/TransT) with sequence-level reinforcement learning. We trained and tested our models on the [GOT10K dataset](http://got-10k.aitestunion.com/), drawing on the implementation from [Kim et al. (2022)](https://arxiv.org/pdf/2208.05810).

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

| Model                      | Data Used       | AO   | SR50  | SR75  |
|----------------------------|-----------------|-------|-------|-------|
| TransT (Baseline)** [1]   | Multiple        | 66.2  | 75.5  | 58.5  |
| SLT + TransT ([1])         | Multiple        | 72.0  | 81.6  | 68.3  |
| SLT + TransT (Ours)        | GOT-10K         | 72.5  | 82.2  | 68.8  |
| **A2C SLT + TransT**       | **GOT-10K**     | **70.5** | **79.9** | **66.3** |

**Notes:**
- *Multiple Dataset* represents a superset of the following publicly available datasets: TrackingNet, GOT-10k, LaSOT, ImageNet-VID, DAVIS, YouTube-VOS, MS-COCO, SBD, LVIS, ECSSD, MSRA10k, and HKU-IS.
- ** Data that we did not verify, but taken directly from Kim et al. (2022) (See Acknowledgments).

![image](https://github.com/user-attachments/assets/00205559-66f0-44b6-8cf5-e8f8fc1348a1)

**Description:** A screenshot of performance comparison of various models on the GOT-10k dataset. The table highlights the Average Overlap (AO), Success Rate at 50% (SR50), and Success Rate at 75% (SR75) metrics for the SLT-enhanced versions (as per the referenced paper and the current work), and the proposed A2C SLT + TransT method.


In the results, it is evident that SLT improves on the Baseline. 
In case of our A2C SLT, we failed to train it for long enough, but it still surpasses the Baseline on all measures and reaches pretty close to the SCST based SLT Tracker.

## Work Yet to be done and further research

- Explore other base trackers (e.g., Siamese trackers, TransDiMP).
- Compare performance with different RL techniques (e.g., PPO, DDPG).
- Add attention mechanisms for smoother tracking.
- Implement real-time tracking capabilities.
- Generate more visual results and user-friendly frontends.


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
