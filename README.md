# ml_demo
Experimental demos of dataset/model configuration and devops. Based on demo from [Sovit Ranjan Rath](https://debuggercafe.com/training-resnet18-from-scratch-using-pytorch/).

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

All Python code lives in pip-installable packages. The semantics for calling scripts is to call, for example, `python training.train` from the root of the folder.

Dockerfiles exist and are used for each relevant step.