print('Importing...')
import pandas as pd
import pickle
from flask import Flask, render_template, request
import json
import sklearn
import sklearn.ensemble._forest
import os

print('loading pkl...')
with open('pickle_files/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('pickle_files/model.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocessing(df, encoders):
    columns = ['Manufacturer', 'Category', 'Screen_Size',
           'RAM', 'GPU', 'Weight', 'Screen_resolution',
           'IPS_Panel', 'Retina_Display', 'Touchscreen', 
           'Quad_HD_plus', 'GHz', 'Intel', 'AMD', 'Storage_Size', 
           'Storage_Size2', 'Storage_Type', 'Storage_Type2', 'os', 
           'Weight_KG']

    df = df[columns]
    
    df1 = df.select_dtypes('object')
    df2 = df.select_dtypes(['int', 'float'])

    for i in df1.columns:
        df1[i] = encoders[i].transform(df1[i])

    x = pd.concat([df1, df2], axis=1)[columns]

    return x

def model_prediction(df1):
    x = preprocessing(df1, encoders)
    return model.predict(x)

columns = ['Manufacturer', 'Category', 'Screen_Size',
           'RAM', 'GPU', 'Weight', 'Screen_resolution',
           'IPS_Panel', 'Retina_Display', 'Touchscreen', 
           'Quad_HD_plus', 'GHz', 'Intel', 'AMD', 'Storage_Size', 
           'Storage_Size2', 'Storage_Type', 'Storage_Type2', 'os', 
           'Weight_KG']

columns_dtypes = {
    'Manufacturer': 'object',
    'Category': 'object',
    'Screen_Size': 'float64',
    'RAM': 'int64',
    'GPU': 'object',
    'Weight': 'object',
    'Screen_resolution': 'object',
    'IPS_Panel': 'int64',
    'Retina_Display': 'int64',
    'Touchscreen': 'int64',
    'Quad_HD_plus': 'int64',
    'GHz': 'float64',
    'Intel': 'int64',
    'AMD': 'int64',
    'Storage_Size': 'object',
    'Storage_Size2': 'object',
    'Storage_Type': 'object',
    'Storage_Type2': 'object',
    'os': 'object',
    'Weight_KG': 'float64',
    'Price': 'float64'
}

with open('uniques.json', 'r') as f:
    uniques = json.load(f)

columns_input_types = {
    'Manufacturer': 'dropdown',
    'Category': 'dropdown',
    'Screen_Size': 'text',
    'RAM': 'text',
    'GPU': 'dropdown',
    'Weight': 'dropdown',
    'Screen_resolution': 'dropdown',
    'IPS_Panel': 'checkbox',
    'Retina_Display': 'checkbox',
    'Touchscreen': 'checkbox',
    'Quad_HD_plus': 'checkbox',
    'GHz': 'float64',
    'Intel': 'checkbox',
    'AMD': 'checkbox',
    'Storage_Size': 'dropdown',
    'Storage_Size2': 'dropdown',
    'Storage_Type': 'dropdown',
    'Storage_Type2': 'dropdown',
    'os': 'dropdown',
    'Weight_KG': 'float64',
    'Price': 'float64'
}

app = Flask(
    __name__,
    template_folder=os.path.join(os.getcwd(), 'templates'),
    static_folder=os.path.join(os.getcwd(), 'static'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        d = {}
        placeholder = {}
        for i in columns:
            placeholder[i] = request.form.get(i)
            if columns_dtypes[i] == 'float64':
                d[i] = float(request.form.get(i))
            elif columns_dtypes[i] == 'int64':
                val = request.form.get(i)
                if val == 'on':
                    d[i] = 1
                elif val == None:
                    d[i] = 0
                else:
                    d[i] = int(val)
            else:
                d[i] = request.form.get(i)
        print(placeholder)
        df = pd.DataFrame(d, index=[0])
        price = f'{round(model_prediction(df)[0]):,}'
    else:
        placeholder = {'Manufacturer': 'Apple', 'Category': 'Ultrabook', 'Screen_Size': '13.3', 'RAM': '8', 'GPU': 'Intel Iris Plus Graphics 640', 'Weight': '1.37kg', 'Screen_resolution': '2560x1600', 'IPS_Panel': 'on', 'Retina_Display': 'on', 'Touchscreen': None, 'Quad_HD_plus': None, 'GHz': '2.3', 'Intel': 'on', 'AMD': None, 'Storage_Size': '128GB', 'Storage_Size2': 'NaN', 'Storage_Type': 'SSD', 'Storage_Type2': 'NaN', 'os': 'macOS', 'Weight_KG': '1.37'}
        price = '-'
    return render_template('index.html', 
                           columns=columns, 
                           price=price,
                           columns_input_types=columns_input_types,
                           uniques=uniques,
                           placeholder=placeholder)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)