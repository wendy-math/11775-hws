#!/bin/python

import numpy
import os
from sklearn.svm.classes import SVC
import cPickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2] #"train_kmeans_$feat_dim_mfcc/"
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]


######## ***--------------------------------------------------------------------*** ########

    Y_truth = []; video_list=[]; groundtruth_label_string = ''
    fopen = open("../all_trn.lst", 'r')
    for line in fopen.readlines():
        splits = line.replace('\n', '').split(' ')
        video_list.append(splits[0])
        if splits[1] == event_name:
            groundtruth_label_string += '1 '
        else:
            groundtruth_label_string += '0 '
    fopen.close()
    Y_truth = numpy.fromstring(groundtruth_label_string.strip(), dtype=int, sep=' ')

    # print 'Totally we get %s labels' % (Y_truth.shape[0])  # for debugging

    # create the feature matrix, in which each row represents a video
    video_num = len(video_list)
    feat_mat = numpy.zeros([video_num, feat_dim])



    for i in xrange(video_num):
        # BOW features of this video
#        if os.path.exists(feat_dir + video_list[i]) == True:
        feat_vec = numpy.genfromtxt(feat_dir + video_list[i], dtype=numpy.float32, delimiter=";")
        assert(feat_vec.shape[0] == feat_dim)
        # fill the feature vector to the matrix
        feat_mat[i,:] = feat_vec

    # initialize svm
    svm = SVC(probability=True)


    # train the svm models
    svm.fit(feat_mat, Y_truth)




    # finally save the k-means model
    cPickle.dump(svm, open(output_file,"wb"), cPickle.HIGHEST_PROTOCOL)


######## ***--------------------------------------------------------------------*** ########

    print 'SVM trained successfully for event %s!' % (event_name)
