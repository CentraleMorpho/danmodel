# danmodel
Implementation of the DAN for head pose estimation

Code intended to predict the yaw-pitch-roll angles relative to the camera from face cropped images.

Please follow the instructions to make it work :

```
python mydataset.py
```
This will create txt files for training and validation sets, containing the relative paths of the images.

```
python mydatasetflop.py
```
This will create a txt file containing the paths of all of the B106 images.

```
python createDictLabels.py
```
This creates the dictionary of keys the images paths and with values the labels, used for getting the labels given a path.

Now you can actually run the training with the DAN and flop loss:

```
python trainDan.py
```

Or run the training with the basic model (one loss function):

```
python trainBasic.py
```



