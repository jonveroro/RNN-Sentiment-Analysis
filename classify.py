
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from keras.models import Sequential
import numpy as np
import pandas as pd
import random


class classify:

	def __init__(self):

		self.model = self.load_model()
		self.dataset_main = self.load_dataset()
		self.max_features = 5000


	def load_dataset(self):
		
		data = pd.read_csv('datasets/combined_neg_pos.csv')
		data = data[['rating','sentence']]
		return data


	def test_data(self):
		
		tokenizer = Tokenizer(nb_words=self.max_features, split=' ')
		tokenizer.fit_on_texts(self.dataset_main['sentence'].values)
		X = tokenizer.texts_to_sequences(self.dataset_main['sentence'].values)
		X = pad_sequences(X)
		Y = pd.get_dummies(self.dataset_main['rating']).values
		X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 10)
		
		# validation_size = 1500
		# batch_size = 32
		# X_validate = X_test[-validation_size:]
		# Y_validate = Y_test[-validation_size:]
		# X_test = X_test[:-validation_size]
		# Y_test = Y_test[:-validation_size]
		# score,acc = self.model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
		# print("score: %.2f" % (score))
		# print("acc: %.2f" % (acc))

		pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

		for x in range(len(X_test)):
			result = self.model.predict(X_test[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)
			
			file = open('files/predict.txt','a+',encoding ='utf-8', errors='ignore')
			file.write('{},{}\n'.format(np.argmax(result),self.dataset_main['sentence'][x]))

		# for x in range(len(X_validate)):
		    
		#     result = self.model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]

		#     if np.argmax(result) == np.argmax(Y_validate[x]):
		#         if np.argmax(Y_validate[x]) == 0:
		#             neg_correct += 1
		#         else:
		#             pos_correct += 1
		       
		#     if np.argmax(Y_validate[x]) == 0:
		#         neg_cnt += 1
		#     else:
		#         pos_cnt += 1



		# print("pos_acc", pos_correct/pos_cnt*100, "%")
		# print("neg_acc", neg_correct/neg_cnt*100, "%")

	def load_model(self):

		json_file = open('Models/Model-Final.json','r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights("Models/Model-Final.h5")
		loaded_model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
		print("Loaded Model from disk")
		return loaded_model


classifier = classify()
classifier.test_data()