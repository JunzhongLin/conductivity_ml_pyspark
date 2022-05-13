from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from udf_data_preparation import nbo_adder
'config("spark.driver.bindAddress", "127.0.0.1")'

csv_file = './/raw_data//RO300 IS NOT NULL SIO2 OVER 20 SI OVER 0 PBO UNDER 30 SLEG CLEANED YEAR OVER 1949 TOTAL 5422 all elements.csv'
spark = (SparkSession
         .builder
         .config("spark.driver.bindAddress", "127.0.0.1")
         .appName('data_reader')
         .getOrCreate())

df = (spark.read.format('csv')
      .option('sep', ';')
      .option('inferSchema', 'true')
      .option('header', 'true')
      .load(csv_file))

df.createOrReplaceTempView('data_tbl')

spark.sql("""select GLASNO fro m data_tbl""").show(5)
