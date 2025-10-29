# EV Smart Range Forecaster

This project predicts the realistic energy consumption (and thus range) of an Electric Vehicle based on real-world factors like speed, weather, and road conditions.

## Features
* Uses a Multi-Variate Linear Regression model trained on the Kaggle EV Energy Consumption Dataset.
* Includes Exploratory Data Analysis (EDA) visualizations.
* (Future Goal: Integrate with a Gen AI chatbot for trip planning).

## Dataset
* [EV Energy Consumption Dataset on Kaggle](https://www.kaggle.com/datasets/ziya07/ev-energy-consumption-dataset)

## How to Run
1. Ensure you have Python and pip installed.
2. Clone the repository.
3. Create and activate a virtual environment: `python3 -m venv env && source env/bin/activate`
4. Install dependencies: `pip install pandas scikit-learn altair jupyterlab`
5. Launch JupyterLab: `jupyter-lab`
6. Open the `EV-Energy-Forecaster.ipynb` notebook.