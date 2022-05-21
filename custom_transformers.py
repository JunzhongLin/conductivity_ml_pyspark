from pyspark.ml import Transformer
from pyspark.sql import DataFrame
import udf_data_preparation as udf_prep
from pyspark.sql.functions import array, col, regexp_replace


class AtomicPropertyTransformer(Transformer):

    def __init__(self, element_df, property_id, num_attrib, col_attrib):
        super(AtomicPropertyTransformer, self).__init__()
        self.element_df = element_df
        self.property_id = property_id
        self.num_attrib = num_attrib
        self.col_attrib = col_attrib

    def _transform(self, df: DataFrame):
        df = df.withColumn(
            '{}'.format(self.property_id),
            udf_prep.atomic_property_adder(
             self.element_df,
             self.property_id,
             self.num_attrib
            )(array(self.col_attrib.to_list()))
        ).select(['*', '{}.*'.format(self.property_id)])
        return df


class ElectronPropertyTransformer(Transformer):

    def __init__(self, electron_prop_dict, property_id, num_attrib, col_attrib, filled_bool):
        super(ElectronPropertyTransformer, self).__init__()
        self.electron_prop_dict = electron_prop_dict
        self.property_id = property_id
        self.num_attrib = num_attrib
        self.col_attrib = col_attrib
        self.filled_bool = filled_bool
        if self.filled_bool:
            self.e_key = property_id + '_filled'
        else:
            self.e_key = property_id + '_unfilled'

    def _transform(self, df: DataFrame):
        df = df.withColumn(
            '{}'.format(self.e_key),
             udf_prep.electron_property_adder(
                 self.electron_prop_dict,
                 self.property_id,
                 self.num_attrib,
                 filled=self.filled_bool
             )(array(self.col_attrib.to_list()))
        ).select(['*', '{}.*'.format(self.e_key)])

        return df
