# DEMO: Prediction of conductivity by pyspark
---
![screenshot](./doc/project icon.jpeg)

# About this REPO
---
This is a demo for the project of Conductivity Prediction based on glass composition. The simplified version of origin
project and codes have been shown here for a demonstration purpose.

# Domain Background & Goal
---
Cover glass have been widely used in electronic devices such as mobile phone, tablet. During the manufacturing process 
or the daily usage of those devices, the static electricity phenomenon may cause some problems. It has been found the 
static electricity phenomenon is correlated with the electrical conductivity of glass itself. A machine learning model 
which can predict the conductivity of glass based on its composition could help with the design of new glass composition 
in the future.

# Data description
---
(insert a image for SciGlass database)

The data used in this project are from both in-house data and public database (e.g. SciGlass). In this demo, only the 
data from sciglass database (http://www.akosgmbh.de/sciglass/sciglass.htm) are used for a demonstration purpose

# Folder Structure

```bash
root/
├── __pycache__
├── configs
│ ├── etl_config.json
│ └── ml_config.json
├── dependencies
│ ├── custom_transformers.py
│ ├── logging_.py
│ ├── spark.py
│ └── udf_data_preparation_decrepit.py
├── derby.log
├── etl_job.py
├── ml_dataset
│ ├── test.parquet
│ └── train.parquet
├── ml_job.py
├── raw_data
│ └── rawdata.csv
├── readme.md
├── rf_model_log
│ └── train_results.txt
├── saved_model
├── schema.json
└── transformed_data.parquet

```

# Technical Details
---
- Feature Engineering

For the feature engineering, additional features related with the electron and atom property has been 
added (e.g. electron orbital occupation, mendeleev number, etc). It is inspired from this scientific publication:
https://www.nature.com/articles/s41598-018-35934-y

- Machine Learning model

Random Forest has been used for a demonstration purpose


# Set-up

(to be added)


