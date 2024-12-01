import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes

df = pd.read_csv('data1.csv')

categorical_columns = ['ip', 'device_type', 'card_type', 'card_status', 'pin_inc_count', 'oper_type']
categorical_data = df[categorical_columns].values

numerical_columns = ['transaction_id', 'device_id', 'tran_code', 'mcc', 'client_id', 'sum', 'balance']
numerical_data = df[numerical_columns].values

data = np.concatenate([numerical_data, categorical_data], axis=1)

categorical_indices = list(range(len(numerical_columns), len(numerical_columns) + len(categorical_columns)))

costs = []

kproto = KPrototypes(n_clusters=15, init='Huang', n_init=7, verbose=2)
clusters = kproto.fit_predict(data, categorical=categorical_indices)
costs.append(kproto.cost_)

df['cluster'] = clusters
print(df.head())
df.to_csv('clustered_data1.csv', index=False)