from pyspark.ml import Transformer
from pyspark.sql import DataFrame
# from udf_data_preparation import atomic_property_adder, electron_property_adder
from pyspark.sql.functions import array, col
from pyspark.sql.functions import pandas_udf, udf
from pyspark.sql.types import DoubleType, IntegerType, FloatType, StructType, StructField, Row
import pyspark.sql.functions as F
import numpy as np
import re
import pickle


def electron_prop_prep(element_df):
    e_conf = element_df['electronic_configuration']
    res_dict = {}
    electron_dict = {'s': 2, 'd': 10, 'p': 6, 'f': 14}
    for property_id in ('s', 'p', 'd', 'f'):
        electron_list_full = []
        for element in e_conf:
            if re.findall(r'{}'.format(property_id), element):
                if re.findall(r'{}\d+'.format(property_id), element):
                    electron_list_full.append(
                        int(re.findall(r'{}\d+'.format(property_id), element)[0][1:])
                    )
                else:
                    electron_list_full.append(1)
            else:
                electron_list_full.append(0)
        electron_list = np.array(electron_list_full)
        unfilled_electron_list = electron_dict.get(property_id)-electron_list
        res_dict[property_id+'_filled'] = electron_list
        res_dict[property_id+'_unfilled'] = unfilled_electron_list

    with open('electron_prop_dict.pickle', 'wb') as f:
        pickle.dump(res_dict, f)


@udf(returnType=DoubleType())
def nbo_udf(Li, Na, K, Rb, Cs, Mg, Ca, Sr, Ba, Al, B, Si):

    NBO_T = (Li+Na+K+Rb+Cs+2*Mg+2*Ca+2*Sr+2*Ba-Al-B)/(Si+Al+B)
    if NBO_T < 0: return 0
    return NBO_T


def atomic_property_adder(element_df, property_id, num_attrib):
    element_series = element_df[property_id].fillna(method='pad').iloc[num_attrib]
    schema = StructType(
        [StructField('{}_max'.format(property_id), FloatType(), False),
         StructField('{}_min'.format(property_id), FloatType(), False),
         StructField('{}_std'.format(property_id), FloatType(), False),
         StructField('{}_mean'.format(property_id), FloatType(), False),
         StructField('{}_mode1'.format(property_id), FloatType(), False),
         StructField('{}_mode2'.format(property_id), FloatType(), False),
         ]
    )

    def atomic_property_udf(composition_list):
        comp_array = np.array(composition_list)
        min_res = element_series.iloc[np.where(comp_array > 0)[0]].min().tolist()
        max_res = element_series.iloc[np.where(comp_array > 0)[0]].max().tolist()
        mean_res = np.matmul(comp_array/100, element_series).tolist()
        std_res = np.matmul(comp_array/100, abs(element_series-mean_res)).tolist()
        mode1_res = element_series.iloc[
            np.where(comp_array == np.sort(comp_array)[-1])[0][0]
        ].tolist()
        mode2_res = element_series.iloc[
            np.where(comp_array == np.sort(comp_array)[-2])[0][0]
        ].tolist()

        # return max_res, min_res, std_res, mean_res, mode1_res, mode2_res
        return (float(max_res), float(min_res), float(std_res), float(mean_res), float(mode1_res), float(mode2_res))
    return udf(atomic_property_udf, schema)


def electron_property_adder(electron_prop_dict, property_id, num_attrib, filled=True):
    if filled:
        e_key = property_id+'_filled'
    else:
        e_key = property_id+'_unfilled'

    electron_series = np.array(electron_prop_dict[e_key])[num_attrib]

    schema = StructType(
        [StructField('{}_max'.format(e_key), FloatType(), False),
         StructField('{}_min'.format(e_key), FloatType(), False),
         StructField('{}_std'.format(e_key), FloatType(), False),
         StructField('{}_mean'.format(e_key) , FloatType(), False),
         StructField('{}_mode1'.format(e_key), FloatType(), False),
         StructField('{}_mode2'.format(e_key), FloatType(), False),
         ]
    )

    def electron_property_udf(composition_list):
        comp_array = np.array(composition_list)
        min_res = electron_series[np.where(comp_array > 0)[0]].min().tolist()
        max_res = electron_series[np.where(comp_array > 0)[0]].max().tolist()
        mean_res = np.matmul(comp_array/100, electron_series).tolist()
        std_res = np.matmul(comp_array/100, abs(electron_series-mean_res)).tolist()
        mode1_res = electron_series[
            np.where(comp_array == np.sort(comp_array)[-1])[0][0]
        ].tolist()
        mode2_res = electron_series[
            np.where(comp_array == np.sort(comp_array)[-2])[0][0]
        ].tolist()

        # return max_res, min_res, std_res, mean_res, mode1_res, mode2_res
        return (float(max_res), float(min_res), float(std_res), float(mean_res), float(mode1_res), float(mode2_res))

    return udf(electron_property_udf, schema)


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
            atomic_property_adder(
             self.element_df,
             self.property_id,
             self.num_attrib
            )(array(self.col_attrib))
        ).select(['*', '{}.*'.format(self.property_id)]).drop(col('{}'.format(self.property_id)))
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
            electron_property_adder(
                 self.electron_prop_dict,
                 self.property_id,
                 self.num_attrib,
                 filled=self.filled_bool
             )(array(self.col_attrib))
        ).select(['*', '{}.*'.format(self.e_key)]).drop(col('{}'.format(self.e_key)))

        return df
