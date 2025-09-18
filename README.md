# RCAL: Reinforced Cross-modal Alignment for Multimodal Sentiment Analysis with Sparse Visual Frames


## Content
- [Code Structure](#Code-structure)
- [Note](#Note)
- [Data Preparation](#Data-preparation)
- [Environment](#Environment)
- [Training](#Training)
- [Citation](#Citation)

## 📁 Code Structure
The repository is organized as follows:
<pre>
├── configs/           # Configuration files for MOSI, MOSEI, CH-SIMS
├── core/              # Core components: dataset loader, scheduler, losses, metrics, etc.
├── models/            # Model definitions: EM-RCAL modules, fusion, gating mechanisms
├── train.py           # Main training + evaluation entry point
├── environment.txt    # Full Conda environment (Python 3.10 + PyTorch + CUDA)
├── yolov5su.pt        # Pretrained YOLOv5 checkpoint
├── data/              # Three different datasets: MOSI, MOSEI, CH-SIMS   
└── README.md          # Project description and usage instructions
</pre>


## 📌 Note

This repository contains the official code for the **EM-RCAL** model, which has been submitted to **NeurIPS 2025** and is currently under review. The paper is available for public review at the following link:

**[EM-RCAL (OpenReview)](https://openreview.net/forum?id=3owTb8JNgJ)**

---

**EM-RCAL** (**E**xplainable **M**emory-Enhanced **R**einforcement-Learned **C**ross-Attention with **A**daptive **L**earning) is a multimodal sentiment analysis model that integrates memory-enhanced representation learning and reinforcement-learned cross-modal attention. This framework is designed for effective sentiment prediction under sparse visual input conditions, making it particularly suitable for real-world multimedia analysis.

Please note that the repository is currently under active development, and the code is subject to change as we continue to improve the model based on reviewer feedback.



## 📁 Data Preparation


The raw data for this project is adapted from the **MMSA** repository, which can be found at the following link:

**[MMSA - Multimodal Sentiment Analysis](https://github.com/thuiar/MMSA)**

After downloading and preprocessing the raw data, organize your data directory as follows:
<pre>
data/
├── mosi/
│   ├── Raw/
│   └── unaligned_50.pkl
├── mosei/
│   ├── Raw/
│   └── unaligned_50.pkl
└── chsims/
    ├── Raw/
    └── unaligned_39.pkl
</pre>

## 🌐 Environment

This project is developed and tested in the following environment:

### **🖥️ Hardware**
- **GPU**: NVIDIA V100-SXM2 (64GB)

### **📦 Key Dependencies**
- **Python**: 3.8.20  
- **PyTorch**: 2.2.1  
- **TorchVision**: 0.17.1  
- **Torchaudio**: 2.2.1  
- **CUDA**: 11.8  
- **cuDNN**: 8.9.2  
- **NVIDIA CUBLAS**: 12.1.3.1  
- **NVIDIA NCCL**: 2.19.3  

---

### **⚠️ Reproducibility Notice**
Training results may vary slightly depending on the hardware configuration. This codebase is optimized for NVIDIA **V100-SXM2** GPUs, but should work on other CUDA-enabled devices with sufficient memory.

---

### **🔗 Full Dependency List**
To recreate the full environment, you can use the provided **`environment.txt`** file:

```bash
conda create --name myenv --file environment.txt
```

## 🚀 Training

This repository provides command-line interfaces for training the **EM-RCAL** model on the **MOSI**, **MOSEI**, and **CH-SIMS** datasets. Simply run the following commands:

```bash
# Train on MOSI
python train.py --config_file configs/mosi.yaml

# Train on MOSEI
python train.py --config_file configs/mosei.yaml

# Train on CH-SIMS
python train.py --config_file configs/chsims.yaml
```


## Citation
