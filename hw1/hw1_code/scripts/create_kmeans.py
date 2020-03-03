#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
import pandas as pd

# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        print "output_file -- path to save the kmeans clustering file"
        exit(1)

######## ***--------------------------------------------------------------------*** ########

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])
    output_path = sys.argv[4]

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model,"rb"))
    print "K-means models loaded successfully!"

    fread = open(file_list, "r")

    for line in fread.readlines():
        mfcc_path = "mfcc/" + line.replace('\n','') + ".mfcc.csv"
        fwrite = open(output_path + line.replace('\n',''),'w')


        cluster_histogram = numpy.zeros(cluster_num)

        if os.path.exists(mfcc_path) == True and os.stat(mfcc_path).st_size > 0:

            array = numpy.genfromtxt(mfcc_path, delimiter=";")
            pred = kmeans.predict(array)

            for x in pred:
                cluster_histogram[x] += 1
            for m in xrange(cluster_num):
                cluster_histogram[m] /= len(pred)

        line = str(cluster_histogram[0])
        for m in range(1, cluster_num):
            line += ';' + str(cluster_histogram[m])

        fwrite.write(line + '\n')
        fwrite.close()

######## ***--------------------------------------------------------------------*** ########

    print "K-means features generated successfully!"
