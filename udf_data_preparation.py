from pyspark.sql.functions import pandas_udf, udf
from pyspark.sql.types import DoubleType, IntegerType, FloatType, StructType, StructField, Row
import pyspark.sql.functions as F
import numpy as np
import re


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

    electron_series = electron_prop_dict[e_key][num_attrib]

    schema = StructType(
        [StructField('{}_max'.format(e_key), FloatType(), False),
         StructField('{}_min'.format(e_key), FloatType(), False),
         StructField('{}_std'.format(e_key), FloatType(), False),
         StructField('{}_mean'.format(e_key), FloatType(), False),
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


if __name__ == '__main__':
    from mendeleev import get_table
    import pickle
    element_df = get_table('elements')

    with open('electron_prop_dict.pickle', 'rb') as f:
        a = pickle.load(f)
