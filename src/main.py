from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess


class FilePaths:
	
	fnCharList = '../model/charList.txt'
	fnAccuracy = '../model/accuracy.txt'
	fnTrain = '../data/'
	fnInfer = '../data/test_images/test-5.png'
	fnCorpus = '../data/corpus.txt'


def infer(model, fnImg):
	
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)
	print('Recognized:', '"' + recognized[0] + '"')
	print('Probability:', probability[0])


def main():
	
	# optional command line args
	parser = argparse.ArgumentParser()
	parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
	parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')

	args = parser.parse_args()

	decoderType = DecoderType.BestPath
	if args.beamsearch:
		decoderType = DecoderType.BeamSearch

	# infer text on test image
	
	print(open(FilePaths.fnAccuracy).read())
	model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
	infer(model, FilePaths.fnInfer)


if __name__ == '__main__':
	main()

