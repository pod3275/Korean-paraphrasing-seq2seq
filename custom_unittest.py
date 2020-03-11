from utils import *

def loader():
	file_name = "paraphrasing data_DH.xlsx"
	dictionary, pair_data = prepareData("kor", file_name)
	ld = Loader(dictionary, pair_data)

	for batch in ld.train:
		print('test for train_loader : ')
		print(batch[0].size(), batch[1].size())
		break
	print('num_batch : {}'.format(len(ld.train)))

	for batch in ld.test:
		print('test for test_loader : ')
		print(batch[0].size(), batch[1].size())
		break
	print('num_batch : {}'.format(len(ld.test)))



if __name__ == '__main__':
	loader()