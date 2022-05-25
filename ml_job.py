'''
ml_job.py

This Python module contains a demo for the machine learning training
job from raw database for glass composition for the prediction of conductivity.
It can be submitted to a Spark cluster (or locally) using the 'spark-submit'
command found in the '/bin' directory of all Spark distributions
(necessary for running any Spark job, locally or otherwise). For
example, this example script can be executed as follows,

    $SPARK_HOME/bin/spark-submit \
    --master spark://localhost:7077 \
    --py-files packages.zip \
    --files configs/etl_config.json \
    jobs/etl_job.py

where packages.zip contains Python modules required by Training job (in
this example it contains a class to provide access to Spark's logger),
which need to be made available to each executor process on every node
in the cluster; ml_config.json is a text file sent to the cluster,
containing a JSON object with all of the configuration parameters
required by the ml job; and, ml_job.py contains the Spark application
to be executed by a driver process on the Spark master node.

For more details on submitting Spark applications, please see here:
http://spark.apache.org/docs/latest/submitting-applications.html

'''

from pyspark.sql.functions import max, min, when, lit, col
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline, PipelineModel
from dependencies.spark import start_spark
from pyspark.ml.evaluation import RegressionEvaluator


def _train_test_split(spark, configs):
    '''
    Write the train set and test set physically onto the disk
    :param spark: Spark Session Object
    :param configs: config dict json
    :return: None
    '''
    # read the transformed data from etl job
    df = spark.read.parquet('transformed_data.parquet')

    # stratified splitting of dataset
    min_ro = df.select(min(df['RO300'])).collect()[0]['min(RO300)']
    max_ro = df.select(max(df['RO300'])).collect()[0]['max(RO300)']
    gaps = [min_ro + (max_ro-min_ro)*i/5 for i in range(6)]
    df = df.withColumn('RO300_stratified_label', when((col('RO300') >= gaps[0]) & (col('RO300') < gaps[1]), '0')
                                                .when((col('RO300') >= gaps[1]) & (col('RO300') < gaps[2]), '1')
                                                .when((col('RO300') >= gaps[2]) & (col('RO300') < gaps[3]), '2')
                                                .when((col('RO300') >= gaps[3]) & (col('RO300') < gaps[4]), '3')
                                                .when((col('RO300') >= gaps[4]) & (col('RO300') <= gaps[5]), '4')
                       )

    fraction = df.select('RO300_stratified_label').distinct()\
                 .withColumn('fraction', lit(1-configs['test_size'])).rdd.collectAsMap()
    train_df = df.stat.sampleBy('RO300_stratified_label', fraction, configs['seed'])
    test_df = df.subtract(train_df)

    # write data into parquet files for the reproducing of ml job
    train_df.drop(col('RO300_stratified_label')).write.mode('overwrite').parquet('ml_dataset/train.parquet')
    test_df.drop(col('RO300_stratified_label')).write.mode('overwrite').parquet('ml_dataset/test.parquet')

    return None


def _extract_data(spark):
    '''
    load train/test set from parquet files
    :param spark: SparkSession object
    :return: dataframe for train and test
    '''

    train_df = spark.read.parquet('./ml_dataset/train.parquet')
    test_df = spark.read.parquet('./ml_dataset/test.parquet')

    return train_df, test_df


def _evaluate_results(configs, predDF):

    regression_evaluator = RegressionEvaluator(
        predictionCol='prediction',
        labelCol='RO300',
        metricName='rmse'
    )
    rmse = regression_evaluator.evaluate(predDF)

    with open(configs['model_log_dir'] + 'train_results.txt', 'w') as writer:
        writer.write("rmse: {}".format(rmse))

    return None


def main():
    '''
    Main ML training job
    :return: None
    '''

    spark, log, configs = start_spark(
        app_name='conductivity_etl_job',
        files=['configs/ml_config.json'])

    train_df, test_df = _extract_data(spark)

    input_cols = train_df.columns[2:20]+train_df.columns[108:]
    vec_assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
    ml_model = RandomForestRegressor(**configs['model_hyperparams'], featuresCol='features',
                                     labelCol='RO300')
    stages = [vec_assembler, ml_model]
    pipe = Pipeline(stages=stages).fit(train_df)

    model_path = configs['model_path']
    log.warn('training job finished')
    pipe.write().overwrite().save(model_path)
    log.warn('Model Saved ...')

    model = PipelineModel.load(model_path)
    pred_df = model.transform(test_df)

    _evaluate_results(configs, pred_df)
    log.warn('Model metrics Saved ...')

    spark.stop()

    return None


if __name__ == '__main__':
    main()
