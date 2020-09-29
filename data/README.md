# Supported datasets

- CamVid
- CityScapes

Note: When referring to the number of classes, the void/unlabeled class is excluded.

## CamVid Dataset

The Cambridge-driving Labeled Video Database (CamVid) is a collection of over ten minutes of high-quality 30Hz footage with object class semantic labels at 1Hz and in part, 15Hz. Each pixel is associated with one of 32 classes.

The CamVid dataset supported here is a 12 class version developed by the authors of SegNet. [Download link here](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid). For actual training, an 11 class version is used - the "road marking" class is combined with the "road" class.

More detailed information about the CamVid dataset can be found [here](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) and on the [SegNet GitHub repository](https://github.com/alexgkendall/SegNet-Tutorial).

## Cityscapes

Cityscapes is a set of stereo video sequences recorded in streets from 50 different cities with 34 different classes. There are 5000 images with fine annotations and 20000 images coarsely annotated.

The version supported here is the finely annotated one with 19 classes.

For more detailed information see the official [website](https://www.cityscapes-dataset.com/) and [repository](https://github.com/mcordts/cityscapesScripts).

The dataset can be downloaded from https://www.cityscapes-dataset.com/downloads/. At this time, a registration is required to download the data.

# Train with your own dataset

Apart from the datasets mentioned above you can train the model with your own data. For that the following prerequisits must be set:

- Directory structure: The directory in which the training, validation and test data lies needs to have the same structure as described in [SegNet dataset structure](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid). Alternatively the folders for training, validation and testing are redefined in the corresponding data-file.
- A data-file is created that defines further details about the dataset. As an example the `camvid.py` and `cityscapes.py` can be taken.
- The masks of the dataset need to be grayscale. Further by default the shades of each category need to correspond to the index of the `color_encoding` variable in the data-file. In the following example *road* would be labeled with the gray value 1.
```python
color_encoding = OrderedDict([('unlabeled', (0, 0, 0)), ('road', (0, 255, 0))])
```
- Add your dataset as argument option to `main.py`. In the following example a model referred to as *kitti* is added.
```python
if args.dataset.lower() == 'camvid':
    from data import CamVid as dataset
elif args.dataset.lower() == 'cityscapes':
    from data import Cityscapes as dataset
elif args.dataset.lower() == 'kitti':
    from data import Kitti as dataset
```

Once the directory, the data-file and the data itself are prepared the model can be trained as described for *camvid* and *cityscapes*. Example for training with KITTI data with only one class apart from *unlabeled*:

```bash
python main.py -m train --batch-size=<batch_size> --with-unlabeled --save-dir save/ENet_KITTI --name enet_kitti --imshow-batch --dataset kitti --dataset-dir dataset/ --width=1242 --height=375
```
