import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import numpy as np

df = pd.read_csv('clean_dataset.csv')

def train_preprocessing(df):
    df = df.replace(np.nan, 'NaN')

    y = df['Price']
    
    df = df[['Manufacturer', 'Category', 'Screen_Size',
             'RAM', 'GPU', 'Weight', 'Screen_resolution',
             'IPS_Panel', 'Retina_Display', 'Touchscreen', 
             'Quad_HD_plus', 'GHz', 'Intel', 'AMD', 'Storage_Size', 
             'Storage_Size2', 'Storage_Type', 'Storage_Type2', 'os', 
             'Weight_KG']]
    
    df1 = df.select_dtypes('object')
    df2 = df.select_dtypes(['int', 'float'])

    encoders = {}
    
    for i in df1.columns:
        encoder = LabelEncoder()
        df1[i] = encoder.fit_transform(df1[i])
        encoders[i] = encoder

    x = pd.concat([df1, df2], axis=1)[['Manufacturer', 'Category', 'Screen_Size',
             'RAM', 'GPU', 'Weight', 'Screen_resolution',
             'IPS_Panel', 'Retina_Display', 'Touchscreen', 
             'Quad_HD_plus', 'GHz', 'Intel', 'AMD', 'Storage_Size', 
             'Storage_Size2', 'Storage_Type', 'Storage_Type2', 'os', 
             'Weight_KG']]
    
    return x, y, encoders

x, y, encoders = train_preprocessing(df)

model = RandomForestRegressor(n_estimators=200)
model.fit(x, y)

if not os.path.exists('pickle_files'):
    print('pickle_files not exist')
    os.mkdir('pickle_files')
else:
    print('pickle_files exist')

with open('pickle_files/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

with open('pickle_files/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model saved')