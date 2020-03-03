import commands
from sklearn import preprocessing
import numpy
import os
from sklearn.svm.classes import SVC
from sklearn.svm import LinearSVC
import cPickle
import sys
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        exit(1)


######## ***--------------------------------------------------------------------*** ########

    model_file = sys.argv[1] #mfcc_pred/svm.$event.$k.model
    feat_dir = sys.argv[2] # val_kmeans_$k/
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    event_name = model_file.split(".")[1]


    Y_truth = []; groundtruth_label_string = ''
    video_list = []
    fopen = open("../all_val.lst", 'r')
    for line in fopen.readlines():
        splits = line.replace('\n', '').split(' ')
        video_list.append(splits[0])
        if splits[1] == event_name:
            groundtruth_label_string += '1 '
        else:
            groundtruth_label_string += '0 '
    fopen.close()
    Y_truth = numpy.fromstring(groundtruth_label_string.strip(), dtype=int, sep=' ')
    svm = cPickle.load(open(model_file,"rb"))

    X = numpy.asarray([])
    for video in video_list:
        feat_vec = numpy.genfromtxt(feat_dir + video, dtype=numpy.float32, delimiter=";")
        assert(feat_vec.shape[0] == feat_dim)
        if len(X) == 0:
            X=[feat_vec]
        else:
            X=numpy.append(X,[feat_vec],axis=0)
    pred = svm.predict_proba(preprocessing.scale(X))


    print model_file.split(".")[1]+" CLASS ACCURACY: "+str(accuracy_score(Y_truth, svm.predict(preprocessing.scale(X))))

    fwrite=open(output_file,"w")
    for i in range(len(pred)):
        fwrite.write(str(pred[i][1])+"\n")
    fwrite.close()

    ######## ***--------------------------------------------------------------------*** ########
