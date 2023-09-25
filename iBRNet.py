import numpy as np
np.random.seed(1234567)
import tensorflow as tf
tf.random.set_seed(1234567)
import random
random.seed(1234567)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.layers import Dropout, AlphaDropout, GaussianDropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras.layers import average
from collections import Counter
from tensorflow.keras.layers import Input
import re, os, csv, math, operator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation

import argparse

parser = argparse.ArgumentParser(description='A test program.')
parser.add_argument("-rlrop", "--reduce_lrop", help="Prints the supplied argument.", default=5, type=int)
parser.add_argument("-es", "--early_stopping", help="Prints the supplied argument.", default=10, type=int)
parser.add_argument("-sm", "--saved_model", help="Prints the supplied argument.", default=None, type=str)
parser.add_argument("-prop", "--property", help="Prints the supplied argument.", default=None, type=str)


args = parser.parse_args()

rlrop = args.reduce_lrop
es = args.early_stopping
sm = args.saved_model
prop = args.property

print("elrop:", rlrop)
print("es:", es)
print("sm:", sm)

#Contains 86 elements (Without Noble elements as it does not forms compounds in normal condition)
elements = ['H','Li','Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe','Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
            'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
            'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er','Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 
            'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu' ]

elements_all = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 
                'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
                'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 
                'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
                'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np',
                'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg',
                'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn']

# Regex to Choose from Element Name, Number and Either of the Brackets
token = re.compile('[A-Z][a-z]?|\d+|[()]')

# Create a dictionary with the Name of the Element as Key and No. of elements as Value
def count_elements(formula):
    tokens = token.findall(str(formula))
    stack = [[]]
    for t in tokens:
        if t.isalpha():
            last = [t]
            stack[-1].append(t)
        elif t.isdigit():
             stack[-1].extend(last*(int(t)-1))
        elif t == '(':
            stack.append([])
        elif t == ')':
            last = stack.pop()
            stack[-1].extend(last)   
    return dict(Counter(stack[-1]))

#Normalize the Value of the Dictionary
def normalize_elements(dictionary):
    dic_val = sum(dictionary.values()) 
    if dic_val == 0:
        factor = 0
    else:    
        factor=1.0/ dic_val  
        
    for k in dictionary:
        dictionary[k] = dictionary[k]*factor
    return dictionary

def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 

print(Diff(elements_all, elements)) 

def elemental_fraction(dataframe):
    print('The loaded dataset has %d entries'%len(dataframe['pretty_comp']))

    #data = mp_data[mp_data['composition'].str.contains('(H|Li|Be|B|C|N|O|F|Na|Mg|Al|Si|P|S|Cl|K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Ac|Th|Pa|U|Np|Pu)[0-9]+', case=True)]
    #data = dataframe[~dataframe['pretty_comp'].str.contains('Bk|Md|Ds|Sg|Ar|No|At|Db|He|Po|Fr|Cm|Cn|Rn|Mt|Fm|Cf|Hs|Ra|Es|Bh|Rf|Lr|Rg|Ne|Am')]
    compounds = dataframe['pretty_comp']

    print('The reduced dataset has %d entries'%len(compounds))
    
    compounds = [count_elements(x) for x in compounds]
    compounds = [normalize_elements(x) for x in compounds]

    in_elements = np.zeros(shape=(len(compounds), len(elements)))
    comp_no = 0

    for compound in compounds:
        keys = compound.keys()
        for key in keys:
            in_elements[comp_no][elements.index(key)] = compound[key]
        comp_no+=1  
    
    data = in_elements
    
    return data

train = pd.read_csv(r'dataset/train.csv') 
val = pd.read_csv(r'dataset/val.csv') 
test = pd.read_csv(r'dataset/test.csv') 
#oqmd = oqmd[~oqmd['formulae'].str.contains('Bk|Md|Ds|Sg|Ar|No|At|Db|He|Po|Fr|Cm|Cn|Rn|Mt|Fm|Cf|Hs|Ra|Es|Bh|Rf|Lr|Rg|Ne|Am', na=False)]

print("done loading")  

prop = '{}'.format(prop)

train = train[train[prop].notnull()]
val = val[val[prop].notnull()]
test = test[test[prop].notnull()]

print(train.shape)
print(val.shape)
print(test.shape)

print(prop)

x_train = train.pop('pretty_comp').to_frame()
y_train = train.pop(prop).to_frame()
x_val = val.pop('pretty_comp').to_frame()
y_val = val.pop(prop).to_frame()
x_test = test.pop('pretty_comp').to_frame()
y_test = test.pop(prop).to_frame()

new_x_train = elemental_fraction(x_train)
new_x_val = elemental_fraction(x_val)
new_x_test = elemental_fraction(x_test)

new_y_train = np.array(y_train)
new_y_val = np.array(y_val)
new_y_test = np.array(y_test)

new_y_train.shape = (len(new_y_train),)
new_y_val.shape = (len(new_y_val),)
new_y_test.shape = (len(new_y_test),)


in_layer = Input(shape=(86,))

layer_1 = Dense(1024)(in_layer)
#layer_1 = GaussianNoise(0.001)(layer_1, training=True)
layer_1 = LeakyReLU()(layer_1)

fcc_1 = Dense(1024)(in_layer)
gsk_1 = add([fcc_1, layer_1])

layer_2 = Dense(1024)(gsk_1)
#layer_2 = GaussianNoise(0.001)(layer_2, training=True)
layer_2 = LeakyReLU()(layer_2)

gsk_2 = add([gsk_1, layer_2])


rayer_1 = Dense(1024)(in_layer)
#rayer_1 = GaussianNoise(0.001)(rayer_1, training=True)
rayer_1 = LeakyReLU()(rayer_1)

rcc_1 = Dense(1024)(in_layer)
rsk_1 = add([rcc_1, rayer_1])

rayer_2 = Dense(1024)(rsk_1)
#rayer_2 = GaussianNoise(0.001)(rayer_2, training=True)
rayer_2 = LeakyReLU()(rayer_2)

rsk_2 = add([rsk_1, rayer_2])


mayer_1 = add([gsk_2, rsk_2])


layer_5 = Dense(512)(mayer_1)
#layer_5 = GaussianNoise(0.001)(layer_5, training=True)
layer_5 = LeakyReLU()(layer_5)

mcc_5 = Dense(512)(mayer_1)
msk_5 = add([mcc_5, layer_5])

layer_6 = Dense(512)(msk_5)
#layer_6 = GaussianNoise(0.001)(layer_6, training=True)
layer_6 = LeakyReLU()(layer_6)

msk_6 = add([msk_5, layer_6])

layer_7 = Dense(512)(msk_6)
#layer_7 = GaussianNoise(0.001)(layer_7, training=True)
layer_7 = LeakyReLU()(layer_7)

msk_7 = add([msk_6, layer_7])

layer_8 = Dense(256)(msk_7)
#layer_8 = GaussianNoise(0.001)(layer_8, training=True)
layer_8 = LeakyReLU()(layer_8)

mcc_8 = Dense(256)(msk_7)
msk_8 = add([mcc_8, layer_8])

layer_9 = Dense(256)(msk_8)
#layer_9 = GaussianNoise(0.001)(layer_9, training=True)
layer_9 = LeakyReLU()(layer_9)

msk_9 = add([msk_8, layer_9])

layer_10 = Dense(256)(msk_9)
#layer_10 = GaussianNoise(0.001)(layer_10, training=True)
layer_10 = LeakyReLU()(layer_10)

msk_10 = add([msk_9, layer_10])

layer_11 = Dense(128)(msk_10)
#layer_11 = GaussianNoise(0.001)(layer_11, training=True)
layer_11 = LeakyReLU()(layer_11)

mcc_11 = Dense(128)(msk_10)
msk_11 = add([mcc_11, layer_11])

layer_12 = Dense(128)(msk_11)
#layer_12 = GaussianNoise(0.001)(layer_12, training=True)
layer_12 = LeakyReLU()(layer_12)

msk_12 = add([msk_11, layer_12])

layer_13 = Dense(128)(msk_12)
#layer_13 = GaussianNoise(0.001)(layer_13, training=True)
layer_13 = LeakyReLU()(layer_13)

msk_13 = add([msk_12, layer_13])

layer_14 = Dense(64)(msk_13)
#layer_14 = GaussianNoise(0.001)(layer_14, training=True)
layer_14 = LeakyReLU()(layer_14)

mcc_14 = Dense(64)(msk_13)
msk_14 = add([mcc_14, layer_14])

layer_15 = Dense(64)(msk_14)
#layer_15 = GaussianNoise(0.001)(layer_15, training=True)
layer_15 = LeakyReLU()(layer_15)

msk_15 = add([msk_14, layer_15])

layer_16 = Dense(32)(msk_15)
#layer_16 = GaussianNoise(0.001)(layer_16, training=True)
layer_16 = LeakyReLU()(layer_16)

mcc_16 = Dense(32)(msk_15)
msk_16 = add([mcc_16, layer_16])

out_layer = Dense(1)(msk_16)

model = Model(inputs=in_layer, outputs=out_layer)

adam = optimizers.Adam(lr=0.0001)
model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=rlrop, min_lr=0.00000001)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=es, restore_best_weights=True)

# Fit the model
model.fit(new_x_train, new_y_train,verbose=2, validation_data=(new_x_val, new_y_val), epochs=3000, batch_size=32, callbacks=[es, reduce_lr])

results = model.evaluate(new_x_test, new_y_test, batch_size=32)
print(results)

model_json = model.to_json()
with open("{}.json".format(sm), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("{}.h5".format(sm))
print("Saved model to disk")