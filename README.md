# Anomaly Detection
Massive real-time data collection requires rapid identification of anomalous observations. However, many current machine learning approaches for anomaly detection require complex, carbon-intensive models to output accuracte predictions. This repository presents models that achieve comparable accuracy but significantly decrease carbon output.

# Getting Started
These examples are run in AzureML via Jupyter Notebook based job submission. Users need to satisfy the following prerequisites and complete the following steps to sucessfully run the examples in `notebooks/`.

## Prerequisites
- AzureML Subscription (to run scripts)
- Terminal / Command Prompt (to setup Azure login)
- Packages (other versions are not guaranteed to work)
  - python == 3.8.5
  - pip == 21.0.1
  - ipython == 7.20.0
  - jupyter-client == 6.1.11

## Setup
1. Create an AzureML workspace ([instructions](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?tabs=azure-portal#create-a-workspace))
3. Create GPU compute instance ([instructions](https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-gpu), NVIDIA Tesla T4 recommended)
2. Create Python environment with prerequisite packages
    - [Install](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) conda environment (suggested)
    - [Install](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) pip virtual environment
    - [Install](https://packaging.python.org/tutorials/installing-packages/) directly onto system via python
4. Clone this repository
5. Run `jupyter notebook` in terminal to open the Jupyter Notebook client
6. Run the examples in `notebooks`
