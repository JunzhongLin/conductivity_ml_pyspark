"""
etl_job.py
~~~~~~~~~~

This Python module contains an example Apache Spark ETL job definition.
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

Our chosen approach for structuring jobs is to separate the individual
'units' of ETL - the Extract, Transform and Load parts - into dedicated
functions, such that the key Transform steps can be covered by tests
and jobs or called from within another environment (e.g. a Jupyter or
Zeppelin notebook).
"""
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
        files=['configs/etl_config.json'])

    # log that main ETL job is starting
    log.warn('etl_job is up-and-running')

    # load schema of db
    with open('schema.json', 'r') as f:
        schema_json = json.load(f)
    schema = StructType.fromJson(schema_json)

    # load config_dict
    if not config:
        with open('configs/etl_config.json', 'r') as f:
            config = json.load(f)

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
        .load('./raw_data/rawdata.csv')
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
    pipe = Pipeline(stages=[atomic_model, electron_model, vec_assembler1])
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
     .parquet('transformed_data.parquet')
     )
    return None


def create_test_data(spark, config):
    """Create test data.

    This function creates both both pre- and post- transformation data
    saved as Parquet files in tests/test_data. This will be used for
    unit tests as well as to load as part of the example ETL job.
    :return: None
    """
    # create example data from scratch
    local_records = [
        Row(id=1, first_name='Dan', second_name='Germain', floor=1),
        Row(id=2, first_name='Dan', second_name='Sommerville', floor=1),
        Row(id=3, first_name='Alex', second_name='Ioannides', floor=2),
        Row(id=4, first_name='Ken', second_name='Lai', floor=2),
        Row(id=5, first_name='Stu', second_name='White', floor=3),
        Row(id=6, first_name='Mark', second_name='Sweeting', floor=3),
        Row(id=7, first_name='Phil', second_name='Bird', floor=4),
        Row(id=8, first_name='Kim', second_name='Suter', floor=4)
    ]

    df = spark.createDataFrame(local_records)

    # write to Parquet file format
    (df
     .coalesce(1)
     .write
     .parquet('tests/test_data/employees', mode='overwrite'))

    # create transformed version of data
    df_tf = transform_data(df, config['steps_per_floor'])

    # write transformed version of data to Parquet
    (df_tf
     .coalesce(1)
     .write
     .parquet('tests/test_data/employees_report', mode='overwrite'))

    return None


# entry point for PySpark ETL application
if __name__ == '__main__':
    main()
