# encoding: utf-8

import json
import sys
import random
import numpy as np
from itertools import chain, izip_longest
from tqdm import tqdm
import learning.linearclassifier as linearclassifier
import colorama; colorama.init(); from colorama import Fore, Style

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

def log_step(s):
	return Style.BRIGHT+Fore.GREEN+"==> "+ Fore.RESET + s + Style.RESET_ALL

print log_step("Assembling feature databases")

with open(sys.argv[1]) as f:
	featurefile = json.load(f)

featuredb = [(class_from_filename(filen), x) for (filen, x) in featurefile.items()]
for c, v in featuredb:
	v.insert(0, 1.0) # prepend 1.0 to every feature vector

vlpdb = filter_class('vlp', featuredb)
diffusedb = filter_class('diffuse', featuredb)

print "total feature db size = {}".format(len(featuredb))
print "  #vlp = {}, #diffuse = {}".format(len(vlpdb), len(diffusedb))

tuning_train = slice_percent(vlpdb, 0, 80) + slice_percent(diffusedb, 0, 80)
tuning_verif = slice_percent(vlpdb, 80, 90) + slice_percent(diffusedb, 80, 90)

eval_train = slice_percent(vlpdb, 0, 90) + slice_percent(diffusedb, 0, 90)
eval_verif = slice_percent(vlpdb, 90, 100) + slice_percent(diffusedb, 90, 100)

print "tuning set:\n  #train={}, #verif={}".format(len(tuning_train), len(tuning_verif))
print "evaluation set:\n  #train={}, #verif={}".format(len(eval_train), len(eval_verif))

print log_step("Finding optimal tuning parameters")

_, l = linearclassifier.find_optimal_weights(np.linspace(0, 5, 100), *(extract_xy(tuning_train) + extract_xy(tuning_verif)))
print "tuning finished!\n-----------\n * λ = {:.4f}".format(l)

print log_step("Evaluating tuned model")

th = linearclassifier.train_weights(l, *extract_xy(eval_train))
print "weight summary\n-----------"
print " * θ[0] = β₀ = {:.4f} (assuming 1-padded feature vectors)".format(th[0])

X_v, Y_v = extract_xy(eval_verif)
errno = linearclassifier.prediction_errors(Y_v, linearclassifier.predict(X_v, th))

print "error summary\n-----------"
print "classification error: #={}, %={}".format(errno, 100.0*errno/len(X_v))
print "by classes:"
vlp_errno = linearclassifier.errors_for_class(encode_class('vlp'), Y_v, linearclassifier.predict(X_v, th))
diffuse_errno = linearclassifier.errors_for_class(encode_class('diffuse'), Y_v, linearclassifier.predict(X_v, th))
print "  vlp->diffuse error: #={}, %={}".format(vlp_errno, 100.0*vlp_errno/len(X_v))
print "  diffuse->vlp error: #={}, %={}".format(diffuse_errno, 100.0*diffuse_errno/len(X_v))
