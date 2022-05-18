import pickle
import json
from mendeleev import get_table
import pandas as pd
import pickle
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col, regexp_replace
import udf_data_preparation as udf_prep
from importlib import reload

'''
## prepare for the raw_df 
element_df = get_table('elements')
e_conf = element_df['electronic_configuration']
file_path = './test_data/test_data.csv'

df = pd.read_csv(file_path, sep=';')
num_attrib = df.columns[28:].astype(int)-1
column_attrib = df.columns[28:]

with open('num_attrib.pickle', 'wb') as out:
    pickle.dump(num_attrib, out)
'''

element_df = get_table('elements')
e_conf = element_df['electronic_configuration']


with open('./num_attrib.pickle', 'rb') as input_file:
    num_attrib = pickle.load(input_file)
with open('./col_attrib.pickle', 'rb') as input_file:
    col_attrib = pickle.load(input_file)

# element_df = element_df.iloc[num_attrib, :]

test_file = './test_data/test_data.csv'

if __name__ == '__main__':

    spark = (SparkSession
             .builder
             .config("spark.driver.bindAddress", "127.0.0.1")
             .appName('data_reader')
             .getOrCreate())

    df = (spark.read.format('csv')
          .option('sep', ';')
          .option('inferSchema', 'true')
          .option('header', 'true')
          .load(test_file))

    for col_name in df.columns:
        if df.select(col(col_name)).dtypes[0][1] == 'string':
            df = (df
                  .withColumn(col_name, regexp_replace(col(col_name), ',', '.')))

        if col_name not in ('GLASNO', 'Year', 'Refer_Id', 'Patent', 'Sleg'):
            df = df.withColumn(col_name, col(col_name).cast('float'))

    test_df = df.select(
        ['*',
         udf_prep.atomic_property_adder(
             element_df,
             'atomic_number',
             num_attrib
         )(array(col_attrib.to_list())).alias('atomic_number_output')])

    # df = (spark.read.format("jdbc")
    #       .option("url", "jdbc:mysql://localhost/conductivity")
    #       .option("driver", "com.mysql.jdbc.Driver")
    #       .option("dbtable", "rawdata")
    #       .option("user", "root")
    #       .option("password", "QIlin618").load())








