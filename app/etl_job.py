"""
etl_job.py
~~~~~~~~~~

This Python module contains a demo for the etl job from raw database for glass composition.
It can be submitted to a Spark cluster (or locally) using the 'spark-submit'
command found in the '/bin' directory of all Spark distributions
(necessary for running any Spark job, locally or otherwise). For
example, this example script can be executed as follows,

    $SPARK_HOME/bin/spark-submit \
    --master spark://localhost:7077 \
    --py-files packages.zip \
    --files configs/etl_config.json \
    jobs/etl_job.py

where packages.zip contains Python modules required by ETL job (in
this example it contains a class to provide access to Spark's logger),
which need to be made available to each executor process on every node
in the cluster; etl_config.json is a text file sent to the cluster,
containing a JSON object with all of the configuration parameters
required by the ETL job; and, etl_job.py contains the Spark application
to be executed by a driver process on the Spark master node.

For more details on submitting Spark applications, please see here:
http://spark.apache.org/docs/latest/submitting-applications.html

"""
# import findspark
# findspark.init()
from pyspark.sql.types import StructType
from pyspark.sql import Row
from pyspark.sql.functions import col, concat_ws, lit

from dependencies.spark import start_spark
import dependencies.custom_transformers as ct
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

import json
from mendeleev import get_table


def main():
    """
    Main ETL script definition.

    :return: None
    """
    # start Spark application and get Spark session, logger and config
    spark, log, config = start_spark(
        app_name='conductivity_etl_job',
        files=['./app/configs/etl_config.json'])

    # log that main ETL job is starting
    log.warn('etl_job is up-and-running')

    # load schema of db
    with open('./app/schema.json', 'r') as f:
        schema_json = json.load(f)
    schema = StructType.fromJson(schema_json)

    # load config_dict
    # if not config:
    #     with open('configs/etl_config.json', 'r') as f:
    #         config = json.load(f)

    # execute ETL pipeline
    data = extract_data(spark, schema)
    data_transformed = transform_data(data, config)
    load_data(data_transformed)

    # log the success and terminate Spark application
    log.warn('test_etl_job is finished')
    spark.stop()
    return None


def extract_data(spark, schema):
    """Load data from csv file format. In production, data will be read from SQL db

    :param spark: Spark session object.
    :return: Spark DataFrame.
    """
    df = (
        spark.read.format('csv')
        .option('sep', ',')
        .option('header', 'true')
        .schema(schema)
        .load('./data/raw_data/rawdata.csv')
    )

    return df


def transform_data(df, config_dict):
    """Transform original dataset.

    :param df: Input DataFrame.
    :param steps_per_floor_: The number of steps per-floor at 43 Tanner
        Street.
    :return: Transformed DataFrame.
    """
    num_attrib = config_dict['num_attrib']
    col_attrib = config_dict['col_attrib']
    input_cols = config_dict['input_cols']
    electron_property_dict = config_dict['electron_property_dict']
    element_df = get_table('elements')

    atomic_model = ct.AtomicPropertyTransformer(
        element_df,
        'mendeleev_number',
        num_attrib,
        col_attrib
    )
    electron_model = ct.ElectronPropertyTransformer(
        electron_property_dict,
        's',
        num_attrib,
        col_attrib,
        filled_bool=True
    )
    vec_assembler1 = VectorAssembler(inputCols=input_cols, outputCol='features')
    pipe = Pipeline(stages=[atomic_model, electron_model])
    pipe_model = pipe.fit(df)

    df_transformed = pipe_model.transform(df)

    return df_transformed


def load_data(df):
    """Collect data locally and write to parquet.

    :param df: DataFrame to print.
    :return: None
    """
    (df
     .write
     .mode('overwrite')
     .parquet('./data/transformed_data.parquet')
     )
    return None


# entry point for PySpark ETL application
if __name__ == '__main__':
    main()

    """
docker run -v $(pwd)/app:/job/app \
-v $(pwd)/data:/job/data \
--env PYSPARK_PYTHON=./environment/bin/python \
--env PYSPARK_DRIVER_PYTHON=python \
-it --name conduct \
-w /job 0c34462d17a8 \
--archives /job/app/conductivity_env.tar.gz#environment \
/job/app/etl_job.py

    """
