#!/usr/bin/env python

# okaeri-kanojo classifier

from __future__ import print_function
import sys
import os
import argparse
import csv
import math
import numpy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset

import play_okaeri

N_IN  = 25 # in
N_OUT = 2  # out
MODEL_FILE = './train.model'

# Network definition
class MLP(chainer.Chain):
    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(n_in, n_units),  # n_in -> n_units
            l2=L.Linear(n_units, n_units),  # n_units -> n_units
            l3=L.Linear(n_units, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

# Test data class
class TestData():
    def __init__(self, test_file):
        self.test_file = test_file

    def convert_to_dataset(self):
        data = []
        target = []
        with open(self.test_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for columns in reader:
                target.append(columns[0])
                data.append(columns[1:])
        # convert to numpy arrays
        data = numpy.array(data, dtype=numpy.float32)
        target = numpy.array(target, dtype=numpy.int32)
        # returns with a dataset format liked by model.predictor()
        return data, target

def main():
    parser = argparse.ArgumentParser(description='okaeri kanojo classifier')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--testfile', '-t', default='',
                        help='Input test file')
    args = parser.parse_args()

    print('# test-file: {}'.format(args.testfile))

    model = L.Classifier(MLP(N_IN, args.unit, N_OUT))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU
    chainer.serializers.load_npz(MODEL_FILE, model)

    # Load the dataset
    test_data = TestData(args.testfile)
    test, label = test_data.convert_to_dataset()

    # Classification
    v = model.predictor(test)
    prediction = numpy.argmax(v.data)
    print("okaeri_kanojo: classification: label=", label[0], "predict=", prediction)

    if label[0] == prediction:
      print("okaeri_kanojo: classification: okaeried")
      play_okaeri.play()
      sys.exit(0)
    else:
      sys.exit(1)

if __name__ == '__main__':
    main()
