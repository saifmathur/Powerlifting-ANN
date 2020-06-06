# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 18:28:37 2019

@author: Saif Mathur
"""

import pandas as pd
df = pd.read_csv('openpowerlifting.csv')

df.isnull()

df['Sex']=df['Sex'].replace('F',0)
df['Sex']=df['Sex'].replace('M',1)


df.drop(["Squat4Kg","Bench4Kg","Deadlift4Kg"],axis=1,inplace=True)

df['Tested']=df['Tested'].replace('No',0)
df['Tested']=df['Tested'].replace('Yes',1)

df = df.dropna()
equip = pd.get_dummies(df['Equipment'])
#df = df.set_index('Date',inplace=True)
#f = pd.value_counts(df.Division, dropna = False)

df.drop(["Equipment"],axis=1,inplace=True)
df.drop(["Event"],axis=1,inplace=True)
df=df.join(equip,how='right')



df = df.drop(["Name","AgeClass","Division","Country","Federation","MeetCountry","MeetState","MeetName",
              "Date"],axis=1)

#df.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15]].values = str.replace('-','+')


df['WeightClassKg'] = df['WeightClassKg'].map(lambda x: x.lstrip('+-A-Z').rstrip('+-A-Z'))



df_g = df[ df['Place'] =='G' ]
df = df.drop(df_g.index, axis=0)

#df = df.set_index('Date',inplace=True)

#not selecting column wraps to avoid dummy trap
x = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
               15,18,19,20,22,23,24,25]].values

y = df.iloc[:,[16,17,21]].values
x = pd.DataFrame(x)
y = pd.DataFrame(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from keras.layers import *
from keras.models import *
import keras
model = Sequential()

model.add(Dense(output_dim = 13,init='uniform',activation='relu',
                input_dim=23))


model.add(Dense(output_dim = 13,init='uniform',activation='relu'))



model.add(Dense(output_dim = 13,init='uniform',activation='relu'))


model.add(Dense(output_dim = 3,init='uniform',activation='softmax'))



model.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])


model.fit(x_train,y_train,batch_size=10,epochs=50)

y_pred = model.predict(x_test)






 
      
       
      
      