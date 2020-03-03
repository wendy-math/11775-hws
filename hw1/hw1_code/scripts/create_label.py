import sys
import numpy

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "open_path -- train/test file"
        exit(1)

    event_name = sys.argv[1]
    open_path = sys.argv[2] #"../all_trn.lst"

######## ***--------------------------------------------------------------------*** ########

    # create train P00x label
    Y_truth = []; video_list=[]; groundtruth_label_string = ''
    fopen = open(open_path, 'r')
    for line in fopen.readlines():
        splits = line.replace('\n', '').split(' ')
        video_list.append(splits[0])
        if splits[1] == event_name:
            groundtruth_label_string += '1 '
        else:
            groundtruth_label_string += '0 '
    fopen.close()
    Y_truth = numpy.fromstring(groundtruth_label_string.strip(), dtype=int, sep=' ')

    if open_path == "../all_trn.lst":
        fwrite = open("../"+event_name+"_train_label", "w")
    elif open_path == "../all_val.lst":
        fwrite = open("../"+event_name+"_val_label", "w")

    for i in range(Y_truth.size):
        fwrite.write(str(Y_truth[i])+"\n")
    fwrite.close()
