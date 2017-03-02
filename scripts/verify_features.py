# encoding: utf-8

import json
import sys
import random
import numpy as np
from itertools import chain, izip_longest
from tqdm import tqdm
import learning.linearclassifier as linearclassifier


def class_from_filename(n):
	if 'vlp' in n:
		return 'vlp'
	elif 'diffuse' in n:
		return 'diffuse'

def encode_class(c):
	return {'vlp': 1, 'diffuse': -1}[c]

def extract_xy(db):
	return np.array(zip(*db)[1]), np.array(map(encode_class, zip(*db)[0]))

def shuffle(a, b):
	return filter(None, chain(*izip_longest(a, b)))

def filter_class(c, db):
	return [x for x in featuredb if x[0] == c]

def slice_percent(x, a, b):
	a_i = int(np.round(a/100.0 * len(x)))
	b_i = int(np.round(b/100.0 * len(x)))
	return x[a_i:b_i]

with open(sys.argv[1]) as f:
	featurefile = json.load(f)

featuredb = [(class_from_filename(filen), x) for (filen, x) in featurefile.items()]
vlpdb = filter_class('vlp', featuredb)
diffusedb = filter_class('diffuse', featuredb)

tuning_train = slice_percent(vlpdb, 0, 80) + slice_percent(diffusedb, 0, 80)
tuning_verif = slice_percent(vlpdb, 80, 90) + slice_percent(diffusedb, 80, 90)

eval_train = slice_percent(vlpdb, 0, 90) + slice_percent(diffusedb, 0, 90)
eval_verif = slice_percent(vlpdb, 90, 100) + slice_percent(diffusedb, 90, 100)


X, Y = extract_xy(trainingdb)
X_v, Y_v = extract_xy(verificationdb)

th, l = linearclassifier.find_optimal_weights(np.linspace(0, 10, 10), X, Y, X_v, Y_v)
print "optimum lambda={}; found theta_hat={}".format(l, th)

errno = linearclassifier.prediction_errors(Y_v, linearclassifier.predict(X_v, th))

print "classification error: #={}, %={}".format(errno, 100.0*errno/len(X_v))
