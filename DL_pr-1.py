import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
import pickle

df=pd.read_excel("heart_2022_no_nans.xlsx",engine="openpyxl")
print(df.head)

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import tensorflow as tf
# from tensorflow import keras
# import keras
# from keras.
# from keras.models import Sequential
# from keras.layers import Dense

# ct_cols = df.select_dtypes(include=['object']).columns.tolist()
# nm_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# print("Categorical Columns:", ct_cols)
# print("Numerical Columns:", nm_cols)


# lable_encode={}
# for col in ct_cols:
#     le=LabelEncoder()
#     df[col]=le.fit_transform(df[col])
#     lable_encode[col]=le
    
# print(lable_encode)

# x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)


# model=Sequential([
#     Dense(32, activation='relu', input_shape=(x_train[1])),
#     Dense(16, activation='relu')
#     Dense(1, actication='sigmoid')

# ]
# )

# model.compile(optimizer='sgd', loss='binary_crossentropy')





