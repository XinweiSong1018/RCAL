# Title



## Content
- [Note](#Note)
- [Data Preparation](#Data-preparation)
- [Environment](#Environment)
- [Training](#Training)
- [Citation](#Citation)


## 📌 Note

This repository contains the official code for the **EM-RCAL** model, which has been submitted to **NeurIPS 2025** and is currently under review. The paper is available for public review at the following link:

**[EM-RCAL (OpenReview)](https://openreview.net/forum?id=3owTb8JNgJ)**

---

**EM-RCAL** (**E**xplainable **M**emory-Enhanced **R**einforcement-Learned **C**ross-Attention with **A**daptive **L**earning) is a multimodal sentiment analysis model that integrates memory-enhanced representation learning and reinforcement-learned cross-modal attention. This framework is designed for effective sentiment prediction under sparse visual input conditions, making it particularly suitable for real-world multimedia analysis.

Please note that the repository is currently under active development, and the code is subject to change as we continue to improve the model based on reviewer feedback.



## Data Preparation


## Environment


## Training

### MOSI
python train.py --config_file configs/mosi.yaml 


### MOSEI
python train.py --config_file configs/mosei.yaml 


### CHSIMS
python train.py --config_file configs/chsims.yaml 


## Citation
