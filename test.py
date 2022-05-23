import pickle
import json
from mendeleev import get_table
import pandas as pd
import pickle
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col, regexp_replace
import dependencies.udf_data_preparation as udf_prep
from importlib import reload
import dependencies.custom_transformers as ct
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType

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


# with open('backup/num_attrib.pickle', 'rb') as input_file:
#     num_attrib = pickle.load(input_file)
# with open('backup/col_attrib.pickle', 'rb') as input_file:
#     col_attrib = pickle.load(input_file)
# with open('backup/input_cols.pickle', 'rb') as input_file:
#     input_cols = pickle.load(input_file)
# with open('backup/electron_prop_dict.pickle', 'rb') as f:
#     electron_property_dict = pickle.load(f)
with open('schema.json', 'r') as f:
    d = json.load(f)

with open('configs/etl_config.json', 'r') as f:
    config_dict = json.load(f)

num_attrib = config_dict['num_attrib']
col_attrib = config_dict['col_attrib']
input_cols = config_dict['input_cols']
electron_property_dict = config_dict['electron_property_dict']

schema = StructType.fromJson(d)

# element_df = element_df.iloc[num_attrib, :]

test_file = './test_data/test_data.csv'
raw_file = './raw_data/rawdata.csv'

if __name__ == '__main__':

    spark = (SparkSession
             .builder
             .config("spark.driver.bindAddress", "127.0.0.1")
             .appName('data_reader')
             .getOrCreate())

    df = (spark.read.format('csv')
          .option('sep', ',')
          .option('inferSchema', 'true')
          .option('header', 'true')
          .load(test_file))

    raw_df = (
        spark.read.format('csv')
         .option('sep', ',')
         .option('header', 'true')
         .schema(schema)
         .load(raw_file)
    )

    '''
    for col_name in df.columns:
        if df.select(col(col_name)).dtypes[0][1] == 'string':
            df = (df
                  .withColumn(col_name, regexp_replace(col(col_name), ',', '.')))

        if col_name not in ('GLASNO', 'Year', 'Refer_Id', 'Patent', 'Sleg'):
            df = df.withColumn(col_name, col(col_name).cast('float'))
    '''


    '''
    ### test for the udf functions
        test_df = df.select(
        ['*',
         udf_prep.atomic_property_adder(
             element_df,
             'mendeleev_number',
             num_attrib
         )(array(col_attrib.to_list())).alias('mendeleev_number'),
         udf_prep.electron_property_adder(
             electron_property_dict,
             's',
             num_attrib
         )(array(col_attrib.to_list())).alias('s_filled'),
         udf_prep.electron_property_adder(
             electron_property_dict,
             's',
             num_attrib,
             filled=False
         )(array(col_attrib.to_list())).alias('s_unfilled')
         ]
    )
    
    '''

    '''
    ### test for custom transformers

    '''
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
    df_example = pipe_model.transform(df)

    # df = (spark.read.format("jdbc")
    #       .option("url", "jdbc:mysql://localhost/conductivity")
    #       .option("driver", "com.mysql.jdbc.Driver")
    #       .option("dbtable", "rawdata")
    #       .option("user", "root")
    #       .option("password", "QIlin618").load())








