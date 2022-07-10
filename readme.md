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
├── Dockerfile
├── LICENSE.md
├── app
│   ├── configs
│   │   ├── etl_config.json
│   │   └── ml_config.json
│   ├── dependencies
│   │   ├── __init__.py
│   │   ├── custom_transformers.py
│   │   ├── logging_.py
│   │   ├── spark.py
│   │   ├── test.py
│   │   └── udf_data_preparation_decrepit.py
│   ├── etl_job.py
│   ├── ml_job.py
│   ├── model_log_dir
│   ├── schema.json
│   └── wordcount.py
├── base
│   ├── Dockerfile
│   ├── README.md
│   └── scripts
│       ├── execute-step.sh
│       ├── finish-step.sh
│       └── wait-for-step.sh
├── data
│   ├── ml_dataset
│   └── raw_data
│       └── rawdata.csv
├── doc
│   └── project icon.jpeg
├── docker-compose.yml
├── docker_test
├── readme.md
├── requirements.txt
└── start-spark.sh
```

# Technical Details
---
- Feature Engineering

For the feature engineering, additional features related with the electron and atom property has been 
added (e.g. electron orbital occupation, mendeleev number, etc). It is inspired from this scientific publication:
https://www.nature.com/articles/s41598-018-35934-y

- Machine Learning model

Random Forest has been used for a demonstration purpose


# Set-up & How to use
---
- General Information

the etl_job.py and ml_job.py inside folder app are the two main demo spark job will be submitted for test.
The wordcount.py in app folder is for basic test purpose. A simple standalone spark cluster will be built as
a testing environment using docker solution. A docker-compose up action will achieve this goal.

The docker compose will create following containers:

container|Exposed ports
---|---
spark-master|9090 7077
spark-worker-1|9091
spark-worker-2|9092

- Installation

The following steps will describe how to install the test environment

- Pre requisites

* Docker installed, Docker-compose installed

- build the base image

the base image contains python and spark. The dockerfile can be found in /base
the dockerfile is obtained by apply some editing the one provided from 
https://dev.to/mvillarrealb/creating-a-spark-standalone-cluster-with-docker-and-docker-compose-2021-update-6l4

```sh
cd ./base
docker build --tag john/pyspark:3.0.2-hadoop3.2-py3.7 .
```

- build the docker image with dependencies

the docker file can be found in the root dir

```sh
docker build --tag conduct-ml-pyspark:0.3 .
```

- compose up the test cluster

the docker-compose yml file can be found in the root dir

```shell
docker-compose up -d
```

- validate the cluster

the webui port of master node can be accessed from localhost:7077


- Binded volumes
- 
local volume will be amounted to the container

Host Mount| Container Mount |Purpose
---|-----------------|---
app| /job/app        |Used to make available your app on all workers & master
data| /job/data       | Used to make available your app's data on all workers & master

- submit the job

log into any node by
```shell
docker exec -it <container-id> /bin/bash
```
submit the job by
```shell
/opt/spark/bin/spark-submit --master spark://spark-master:7077 --files /job/app/configs/etl_conf
ig.json /job/app/etl_job.py
```

to change the deploy mode, please go to the etl_job.py
modify the master argument for the start_spark function




