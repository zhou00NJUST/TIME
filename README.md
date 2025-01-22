# TIME: Trajectory and Interaction-based Memory Enhancement Framework

## Overview

This repository contains the official implementation of paper **"TIME: Trajectory and Interaction-based Memory Enhancement Framework for Multi-Agent Trajectory Prediction"**. 

Our motivation is to address the limitations of existing parameter-based methods in multi-agent trajectory prediction, including their lack of sensitivity to special samples and limited interpretability in multi-modal trajectory prediction. Inspired by how humans recall past experiences to aid decision-making, we propose a memory-enhanced framework that explicitly stores and retrieves relevant information to improve prediction accuracy and generalization.

---

## Code Structure

```plaintext
.
├── basemodel.py                # Base model definitions
├── causal_disentanglement.py   # Feature disentanglement module
├── class2num.py                # Class-to-number mappings
├── common.py                   # Shared utilities
├── diffusion.py                # Diffusion processes
├── keep_functions.py           # Supportive functions
├── models.py                   # Model architectures
├── my_decoder.py               # Custom decoder implementation
├── my_tcns.py                  # Temporal convolutional network
├── Processor.py                # Main processing logic
├── test_ethucy.sh              # Test script for ETH/UCY dataset
├── test_sdd.sh                 # Test script for SDD
├── train.py                    # Training pipeline
├── train_ethucy.sh             # Train script for ETH/UCY
├── train_sdd.sh                # Train script for SDD
├── transformers_.py            # Transformer-based modules
├── utils_ethucy.py             # Utilities for ETH/UCY dataset
├── utils_sdd.py                # Utilities for SDD dataset
```

---

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Unzip the provided `data.zip` file into the current directory to set up the datasets. This will create the following structure:

```plaintext
./data
├── eth      # ETH dataset
├── ucy      # UCY dataset
├── sdd      # Stanford Drone Dataset
```

Use this structure to ensure correct dataset paths during training and testing.

### Training

Run the following scripts to train on respective datasets:

```bash
bash train_ethucy.sh  # For ETH/UCY
bash train_sdd.sh     # For SDD
```

### Testing

Use the test scripts for evaluation:

```bash
bash test_ethucy.sh  # For ETH/UCY
bash test_sdd.sh     # For SDD
```
