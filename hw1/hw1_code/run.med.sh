
#!/bin/bash

# An example script for multimedia event detection (MED) of Homework 1
# Before running this script, you are supposed to have the features by running run.feature.sh

# Note that this script gives you the very basic setup. Its configuration is by no means the optimal.
# This is NOT the only solution by which you approach the problem. We highly encourage you to create
# your own setups.

# Paths to different tools;
opensmile_path=/home/ubuntu/tools/openSMILE-2.1.0/bin/linux_x64_standalone_static
speech_tools_path=/home/ubuntu/tools/speech_tools/bin
ffmpeg_path=/home/ubuntu/tools/ffmpeg-2.2.4
map_path=/home/ubuntu/tools/mAP
export PATH=$opensmile_path:$speech_tools_path:$ffmpeg_path:$map_path:$PATH
export LD_LIBRARY_PATH=$ffmpeg_path/libs:$opensmile_path/lib:$LD_LIBRARY_PATH

for feat_dim_mfcc in 50 100 150 200; do
echo "#####################################"
echo "#       MED with MFCC Features      #"
echo "#####################################"
mkdir -p mfcc_pred
# iterate over the events

for event in P001 P002 P003; do
    echo "=========  Event $event, k = $feat_dim_mfcc  ========="
  # create training/test labeling
  python2 scripts/create_label.py $event "../all_trn.lst" || exit 1;
  python2 scripts/create_label.py $event "../all_val.lst" || exit 1;

  # now train a svm model
  python2 scripts/train_svm.py $event "train_kmeans_"$feat_dim_mfcc"/" $feat_dim_mfcc mfcc_pred/svm.$event.$feat_dim_mfcc.model || exit 1;

  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred
  python2 scripts/test_svm.py mfcc_pred/svm.$event.$feat_dim_mfcc.model "val_kmeans_"$feat_dim_mfcc"/" $feat_dim_mfcc mfcc_pred/${event}.$feat_dim_mfcc.pred || exit 1;

  # compute the average precision by calling the mAP package
  ap ../${event}_val_label mfcc_pred/${event}.$feat_dim_mfcc.pred
done
done


echo ""
echo "#####################################"
echo "#       MED with ASR Features       #"
echo "#####################################"
mkdir -p asr_pred
# iterate over the events
#feat_dim_asr=983
feat_dim_asr=8197
for event in P001 P002 P003; do

echo "=========  Event $event  ========="
# now train a svm model
  python2 scripts/train_svm.py $event "asrfeat/" $feat_dim_asr asr_pred/svm.$event.model || exit 1;

# apply the svm model to *ALL* the testing videos;
# output the score of each testing video to a file ${event}_pred
  python2 scripts/test_svm.py asr_pred/svm.$event.model "asrfeat/" $feat_dim_asr asr_pred/${event}_asr.pred || exit 1;

# compute the average precision by calling the mAP package
ap ../${event}_val_label asr_pred/${event}_asr.pred
done
















#!/bin/bash

# An example script for multimedia event detection (MED) of Homework 1
# Before running this script, you are supposed to have the features by running run.feature.sh

# Note that this script gives you the very basic setup. Its configuration is by no means the optimal.
# This is NOT the only solution by which you approach the problem. We highly encourage you to create
# your own setups.


##### # Paths to different tools;
##### map_path=/home/ubuntu/tools/mAP
##### export PATH=$map_path:$PATH

##### echo "#####################################"
##### echo "#       MED with MFCC Features      #"
##### echo "#####################################"
##### mkdir -p mfcc_pred
##### # iterate over the events
##### feat_dim_mfcc=200
##### for event in P001 P002 P003; do
#####   echo "=========  Event $event  ========="
#####   # now train a svm model
#####   python2 scripts/train_svm.py $event "train_kmeans/" $feat_dim_mfcc mfcc_pred/svm.$event.model || exit 1;
#####   # apply the svm model to *ALL* the testing videos;
#####   # output the score of each testing video to a file ${event}_pred
#####   python2 scripts/test_svm.py mfcc_pred/svm.$event.model "val_kmeans/" $feat_dim_mfcc mfcc_pred/${event}_mfcc.lst || exit 1;
#####   # compute the average precision by calling the mAP package
#####   ap list/${event}_val_label mfcc_pred/${event}_mfcc.lst
##### done

##### echo ""
##### echo "#####################################"
##### echo "#       MED with ASR Features       #"
##### echo "#####################################"
##### mkdir -p asr_pred
##### # iterate over the events
##### feat_dim_asr=983
##### for event in P001 P002 P003; do
##### echo "=========  Event $event  ========="
##### # now train a svm model
##### python2 scripts/train_svm.py $event "asrfeat/" $feat_dim_asr asr_pred/svm.$event.model || exit 1;
##### # apply the svm model to *ALL* the testing videos;
##### # output the score of each testing video to a file ${event}_pred
##### python2 scripts/test_svm.py asr_pred/svm.$event.model "asrfeat/" $feat_dim_asr asr_pred/${event}_asr.lst || exit 1;
##### # compute the average precision by calling the mAP package
##### ap list/${event}_val_label asr_pred/${event}_asr.lst
##### done
