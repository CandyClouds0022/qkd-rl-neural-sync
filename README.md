# RL-Optimized Quantum Key Distribution (QKD) with Neural Synchronization

This repository stores a simulation framework featuring an RL training environment, developed for a Master's thesis on QKD protocols modelling. The project, thus, contains a physical model of an advanced version of BB84, whose keys are exchanged via Tree Parity Machines (TPMs) and adaptively optimized at sifting step with a PPO agent. The model is also meant to be compared with a simplified version of the original BB84 scheme, implemented to verify theoretical, expected results.

---

## Repository structure

The codebase is organized into modular Python scripts:

* `physicalengine.py` – The file provides the baseline physical simulation of the quantum channel and standard BB84 protocol.
* `QKDengine_advanced.py` – The script depicts an extended QKD physical engine accounting for the following: fiber attenuation ($dB/km$), AWGN, dark count rates, photon number splitting (PNS) eavesdropping (Eve) attacks, Error Correction (Binary Entropy Leakage approach) and Privacy Amplification.
* `tpm_neural.py` – Implementation of Tree Parity Machines ($K=3, N=4, L=6, M=3, B=2$) for neural synchronization and shared key generation is portraied in the homonymous file.
* `environment.py` – This implements a Gymnasium Environment (`QKDEnv`), whose role is to interface the physical simulation and TPM logic with standard RL observation/action spaces.
* `train_QKDenv.py` – The script illustrates a PPO agent training pipeline employing Stable-Baselines3, featuring diagnostic callbacks and checkpoint saving.
* `test_QKDenv.py` – This is an evaluation script, designed to test trained RL policies across varying channel distances ($0–150 \text{ km}$) and log some useful metrics to a dedicated CSV.
* `train_dummy.py` – The file builds up a benchmark script, used for training a static baseline model (`MLPRegressor`) on fixed synthetic QKD datasets produced by the baseline model simulation on a finite set of distances.

---

## Fundamental requirements and installation guideline

### 1. Cloning the repository:
```bash
git clone https://github.com/your-username/qkd-rl-neural-sync.git
cd qkd-rl-neural-sync
```

### 2. Setting up the virtual environment:
*Usage of Python 3.10+ is strongly recommended.*
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or: venv\Scripts\activate  # On Windows
```

### 3. Installing dependencies:
```bash
pip install -r requirements.txt
```


## Complete usage guideline
Note: commented files are not supposed to be executed individually, since already imported by those directly using their structures. However, they are listed below to feature the correct workflow of the models design.
To carry out the thesis experiments, files are required to be executed in the following order:

### 1. Generate the static dataset and train the benchmark MLP model:
```bash
#python physicalengine.py
python train_dummy.py
```

### 2. Create the advanced model and environment, with TPM interface:
```bash
#python QKDengine_advanced.py
#python tpm_neural.py
#python environment.py
```

### 3. Train the PPO agent, then, test its acquired learning competencies:
```bash
python train_QKDenv.py
python test_QKDenv.py
```

## Experiments and comparisons are left to user design.