# ml_demo
Experimental demos of dataset/model configuration and devops.

# Installation

```
conda env create -f environment.yml
conda activate ml_demo
pip install -e .
```

# Usage Demo

**Train a model**

```
timnet train
```

**View model runs in mlflow**

```
mlflow server --host 0.0.0.0
# go to http://localhost:5000
```

**View model runs in AIM-MLFlow**

```
# Make AIM repo
mkdir ~/aim_repo
# Bring up AIM server
cd ~/aim_repo
aim up
# Sync AIM and MLFlow
aimlflow sync --mlflow-tracking-uri=http://localhost:5000 --aim-repo=~/aim_repo
```

# Organization:
- Training Repo (this repo)
	- Model definitions
	- Dataset preparation and collation
	- Config
	- Training
	- Metrics
    - Scripts
- Inference Repo
	- ROS Runtime

# Design Principles:
All config is in a hierarchical folder of YAML files, similar to Hydra.

Datasets and models are backed up remotely and stored, versioned, and accessed through MLFlow.

During training, progress is visualized through the AIM frontend.

Where applicable, functions in dataset preparation, training, and metrics have unit tests written in those folders.

Code is commented with Google-style docstrings and type annotation.

Whenever a script is executed whose products are expected to persist, the state of the entire training repo, and all relevant configuration including arguments, are stored as an artifact of the script execution. This applies to dataset preparation, training, and fine tuning.

All Python code lives in pip-installable packages. The semantics for calling scripts is to call, for example, `timnet train` from the ml_demo conda env.

Dockerfiles exist and are used for each relevant step.