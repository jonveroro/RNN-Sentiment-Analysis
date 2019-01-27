
import re
import random
from os import listdir
from os.path import isfile, join


class dataset_creator:

	def __init__(self):
		self.stop_words = self.load_stopwords()

	def load_stopwords(self):
		file = open('files/stop-word-list.csv','r')
		return file.read().split('\n')

	def main(self):

		dataset_main = []

		path_list = ['files/neg/','files/pos/']

		for path in path_list:

			file_list = [f for f in listdir(path) if isfile(join(path, f))]
			file_list = file_list[0:5000]

			for file in file_list:

				file_path = open('{}{}'.format(str(path),str(file)),'r',encoding ='utf-8', errors='ignore')
				file_content = file_path.read()
				file_name = file.replace('.txt','')
				rating = file_name.split('_')[1]
				paragraph = self.clean_paragraph(file_content)
				dataset_main.append([rating,paragraph])


		random.shuffle(dataset_main)
		dataset_file = open('datasets/combined_neg_pos.csv','a+',encoding ='utf-8', errors='ignore')
		dataset_file.write('rating,sentence\n')
		for row in dataset_main:
			dataset_file.write('{},{}\n'.format(row[0],row[1]))

		dataset_file.close()
		print('Done.')

	def clean_paragraph(self,paragraph):
		symbols = [	'`','~','!','@','#','$','%','^','&','*','(',')','_','-',
					'+','=','{','[','}','}','|','\',<',',','>','.','?','/',
					',',"'",'``','\\\\','--','1','2','3','4','5','6','7','8'
					,'9','0','\\',"''","'''",'\n'
				]
		word_list = []
		string_filter = r'[^\w\s]*'
		words = paragraph.split(' ')
		new_word_list = []

		for word in words:
			word = word.lower()
			word = re.sub(string_filter,'',word)
			if word not in symbols:
				word = re.sub('[^a-zA-z0-9\s]','',word)
				word_list.append(word)

		paragraph = ''
		for i in range(len(word_list)):

			if i != len(word_list)-1:
				paragraph += '{} '.format(word_list[i])
			else:
				paragraph += '{}'.format(word_list[i])

		return paragraph

create = dataset_creator()
create.main()