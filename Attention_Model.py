
import numpy as np

from keras.models import Model
from keras.layers import Input,Dense, merge, RepeatVector, Permute
from keras.layers import TimeDistributed,Reshape,Flatten,Lambda
import keras.backend as K



inputsize=4

InputDim=5
LenghtOfSeq=3
num_classes=7

yi = Input(shape=(LenghtOfSeq, InputDim, ))
yiw=TimeDistributed(Dense(InputDim,activation= 'linear'), input_shape=(LenghtOfSeq, InputDim))(yi)


c_input=Input(shape=(InputDim,))
c_repeated=RepeatVector(LenghtOfSeq)(c_input)
cw=TimeDistributed(Dense(InputDim,activation= 'linear'), input_shape=(LenghtOfSeq, InputDim))(c_repeated)

#merged_vector=yi*w+c*w

merged_vector=merge([yiw,cw],mode='sum')

m = TimeDistributed(Dense(1, activation='tanh'))(merged_vector)

m=Flatten()(m)

weights=Dense(LenghtOfSeq,activation='softmax')(m)


d = yi._keras_shape[2]

weights_matrix = RepeatVector(d)(weights)

weights_matrix = Permute((2, 1))(weights_matrix)
si_yi = merge([yi, weights_matrix], mode='mul')


z = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(si_yi)


F1=Dense(20,activation='relu')(z)
F2=Dense(20,activation='relu')(F1)
F3=Dense(num_classes,activation='softmax')(F2)




model1 = Model(input=[yi,c_input] ,output=weights)
model1.compile(optimizer='rmsprop',loss='mse', metrics=['accuracy'])
model2 = Model(input=[yi,c_input] ,output=F3)
model2.compile(optimizer='rmsprop',loss='mse', metrics=['accuracy'])




##model3 = Model(input=[yi,c_input] ,output=cw)
##model3.compile(optimizer='rmsprop',loss='mse', metrics=['accuracy'])
##
##model4 = Model(input=[yi,c_input] ,output=yiw)
##model4.compile(optimizer='rmsprop',loss='mse', metrics=['accuracy'])
##
##model5 = Model(input=[yi,c_input] ,output=merged_vector)
##model5.compile(optimizer='rmsprop',loss='mse', metrics=['accuracy'])


# Generate Data
n_samples=23
inputdata= np.random.rand(n_samples,LenghtOfSeq,InputDim)
c_context=np.random.rand(n_samples,InputDim)
#******************************************
labels=np.array( [f for f in range(num_classes)]*3+[0,1]) # num_classes =7. then  7*3+2 =23 labels of values[0..6] 

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
# convert integers to dummy variables (i.e. one hot encoded)
labels = np_utils.to_categorical(encoded_Y)
#labels=np.expand_dims(labels, axis=1)

#**********************************************

model2.fit([inputdata ,c_context],labels, nb_epoch=200)
y=model1.predict([inputdata ,c_context]) 

sample=np.expand_dims(inputdata[0],axis=0)

##W_context=model3.predict([sample ,c_context])
##W_Y=model4.predict([sample ,c_context])
##Merged_V=model5.predict([sample ,c_context])
##
##print Merged_V-W_context-W_Y

