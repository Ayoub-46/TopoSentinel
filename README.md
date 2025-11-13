# TopoSentinel

A framework for benchmarking backdoor attacks and defenses in Federated Learning, featuring a novel defense mechanism based on Topological Data Analysis (TDA).

## Overview

Federated Learning (FL) enables collaborative model training without sharing raw data, but it is vulnerable to malicious clients who can inject backdoors into the global model. These backdoors cause the model to misclassify specific inputs (e.g., an image with a small patch) while maintaining high accuracy on the main task, making them difficult to detect.

**TopoSentinel** is a PyTorch-based framework designed to:
1.  **Simulate** a variety of state-of-the-art backdoor attacks in an FL setting (e.g., A3FL, IBA, Neurotoxin).
2.  **Benchmark** existing defense mechanisms (e.g., Krum, Flame, DeepSight).
3.  **Introduce** a novel defense, `TopologicalBiasDefenseServer`, which uses Topological Data Analysis to detect anomalous client updates by analyzing their topological "shape" and statistical bias.

## Core Idea: The Topological Defense

The primary contribution of this repository is the `TopologicalBiasDefenseServer` (found in `src/defenses/tda_bias_defense.py`). This defense operates on the principle that malicious client updates, which are crafted to inject a backdoor, will alter the geometric and topological structure of the model's parameter space in a detectable way.

It employs a two-stage detection strategy:

1.  **Inter-Round TDA Trigger:** The server analyzes the topological structure of *client bias vectors* from the current round. It computes a **persistence diagram** (using `persistent_homology/analyzer.py`) and measures its distance (using the **bottleneck distance**) from the previous round's diagram. A sudden, large change in topology (distance exceeding a decaying threshold) indicates a coordinated attack and triggers the filtering mechanism.

2.  **Intra-Round Bias Filtering:** When triggered, the server identifies outliers by:
    * Calculating the distance (e.g., Euclidean or cosine) of each client's bias vector from the round's median bias vector.
    * Filtering out clients whose distance falls outside a dynamically learned "benign interval," which is based on the historical distribution of distances from previous benign rounds.

This hybrid approach aims to be robust against stealthy attacks that may not be detectable by simple distance metrics alone.

## Features

This framework is modular and easily extensible. It includes implementations of:

### 1. Backdoor Attacks
* **A3FL** (`a3fl_client.py`): Adversarial Adaptive Anchor attack.
* **IBA** (`iba_client.py`): Irreversible Backdoor Attack using a generative trigger.
* **Neurotoxin** (`neurotoxin_client.py`): A stealthy attack that constrains updates to less important parameters.
* **TDFed** (`tdfed_client.py`): A three-stage defense-aware attack.
* **Triggers:** Includes static `PatchTrigger`, dynamic `A3FLTrigger`, and generative `IBATrigger`.

### 2. Aggregation Defenses
* **TopoSentinel (Ours)**: `TopologicalBiasDefenseServer` and `AnalysisServer` (for observation).
* **Krum / Multi-Krum**: `MKrumServer`.
* **Flame**: `FlameServer`.
* **DeepSight**: `DeepSightServer`.
* **Norm Clipping**: `NormClippingServer` (can be combined with Differential Privacy).

### 3. Datasets
* **CIFAR-10** (`cifar10.py`)
* **GTSRB** (`gtsrb.py`)
* **FEMNIST** (`femnist.py`)
* **MNIST** (`mnist.py`)
* **ImageNet** (`imagenet.py`) (and TinyImageNet)

## Project Structure
TopoSentinel/ 
├── main.py                # Main experiment entry point 
├── requirements.txt       # Python dependencies 
└── src/ 
    ├── attacks/           # Implementations of backdoor attacks (A3FL, IBA, etc.) 
    ├── defenses/          # Implementations of defense mechanisms (Krum, Flame, TDA-Bias) 
    ├── datasets/          # Data loaders and adapters (CIFAR10, GTSRB, etc.) 
    ├── experiment/ 
    │   ├── configs/       # YAML configuration files for experiments 
    │   ├── runner.py      # Serial experiment runner 
    │   └── parallel_runner.py # Parallel experiment runner 
    ├── fl/                # Core FL client/server logic (FedAvg, FedProx) 
    ├── models/            # Model architectures (ResNet, CNNs) 
    └── persistent_homology/ # TDA utilities (analyzer, metrics) 

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/ayoub-46/toposentinel.git](https://github.com/ayoub-46/toposentinel.git)
    cd toposentinel
    ```

2.  Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run an Experiment

All experiments are driven by YAML configuration files located in `src/experiment/configs/`.

1.  **Choose a Configuration:**
    Select a config file based on the attack and defense you wish to test. For example, to test the **TDA-Bias defense** against the **A3FL attack** on the **GTSRB dataset**, you would use `src/experiment/configs/tda/a3fl_analysis_gtsrb.yml`.

2.  **Run the Experiment:**
    Execute `main.py` and pass the path to your chosen configuration file.

    ```bash
    python main.py --config src/experiment/configs/tda/a3fl_analysis_gtsrb.yml
    ```

3.  **Run in Parallel:**
    For faster execution, you can parallelize client training using the `--parallel` flag. This will use the `ParallelFederatedExperiment` runner.

    ```bash
    python main.py --config src/experiment/configs/tda/a3fl_analysis_gtsrb.yml --parallel
    ```

4.  **Check Results:**
    Metrics, including main model accuracy and backdoor attack success rate (ASR), will be logged to a CSV file in the `results/` directory, named according to the `experiment_name` in the config file.

### Customizing Experiments

You can easily define new experiments by creating a new `.yml` file. The key sections are:

* `data_params`: Specify the `dataset_name` (e.g., 'cifar10', 'gtsrb') and `strategy` (e.g., 'iid', 'dirichlet').
* `fl_params`: Control the federated learning setup (`num_clients`, `clients_per_round`, `num_rounds`).
* `attack_params`: Set `enabled: true`, choose the `name` (e.g., 'a3fl', 'neurotoxin'), and define `malicious_client_ids` and other attack-specific parameters.
* `defense_params`: Set `enabled: true`, choose the `name` (e.g., 'tda_bias', 'krum', 'flame'), and set its hyperparameters.
