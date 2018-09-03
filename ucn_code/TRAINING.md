
# UCN training

```
# set the caffe path in __init__paths.py

```

```
# set the following config variables in lib/config.py

NET_PROTOTXT = './nets/train.prototxt'
PRETRAINED_MODEL = './nets/bvlc_googlenet/bvlc_googlenet.caffemodel'
CLASSIFIER = False
CLASS_NAME = any one of 'chair'/'sofa'/'bed'/'diningtable'
```

```
# run the following to train the UCN
python run_main.py
```
# Viewpoint Classifier Training:

After having trained the UCN, set the following configs in the file lib/config.py, to train the Viewpoint Classifier Network
```
# set the following config variables in lib/config.py

NET_PROTOTXT = './nets/train_classifier.prototxt'
PRETRAINED_MODEL = '/path/to/trained/ucn_model'
CLASSIFIER = True
CLASS_NAME = any one of 'chair'/'sofa'/'bed'/'diningtable'
```

```
# run the following to train the classifier
python run_main.py
```

# UCN Hyperparameters
The hyperparameters used, have been set in the file lib/config.py.

_Note: Please look into the comments in lib/config.py for additional information__
