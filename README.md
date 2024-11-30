This codebase was created as a submission for CSCE-642 at Texas A&M University by Dineth Gunawardena and Wahib Kapdi.

# Introduction

Most visual object trackers use Frame Level Tracking. But this does not account for moving and intermediary frames with may have obstacles blocking the subject, motion blur or ambient occlusion. So our project aim to treat this problem as a sequential problem and use the frames and bounding boxes generated in the past as an input to our RL network.

# Improving Object Tracking Networks using Deep Reinforcement Learning

The goal of this project is to improve the existing tracking networks using Reinforcement Learning Algorithms. 
In this project we use Transformer Tracking or [TransT](https://github.com/chenxin-dlut/TransT) coupled with REINFORCE learning to improve performance on Visual Object Tracking Datasets. We use the [GOT10K](http://got-10k.aitestunion.com/) dataset for the same. Using the implementation suggested by [Kim, Minji et. al.](https://arxiv.org/pdf/2208.05810).

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
