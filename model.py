from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

df = pd.read_csv('clustered_data1copy.csv')
df2 = pd.read_csv('data2.csv')

categorical_columns = ['ip', 'device_type', 'card_type', 'card_status', 'pin_inc_count', 'oper_type']

numerical_columns = ['transaction_id', 'device_id', 'tran_code', 'mcc', 'client_id', 'sum', 'balance']

X_train = df[numerical_columns + categorical_columns]
y_train = df['cluster']

X_test = df2[numerical_columns + categorical_columns]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=300, random_state=10))
])

model.fit(X_train, y_train)

cluster_pred = model.predict(X_test)

df2['predicted_cluster'] = cluster_pred

df2.to_csv('pred_data.csv', index=False)