import os
import values as v

import numpy as np
import pandas as pd


def get_refs():
    ref_path = os.path.join(v.data_path, 'refs.csv')
    table = pd.read_csv(ref_path)

    return {c: table['ref_name'][table['cluster'] == c].values for c in table['cluster'].unique()}


if __name__ == '__main__':
    get_refs()


