## Lighthead - RCNN

Implementation of [Lighthead-RCNN](https://arxiv.org/abs/1711.07264) in Pytorch 0.4.1 codes

## Contributions

- The first reimplementation of [Lighthead-RCNN](https://arxiv.org/abs/1711.07264)  in full stable Pytorch codes

- A new implementation of CUDA codes of [Position Sensitive ROI Pooling](https://arxiv.org/abs/1605.06409) is provided, in case all previous ones don't work under the Pytorch version >= 0.4

- Pretrained model is provided

- ------

  At the time being, this repo only support single batch training.

- Dataloader from chainercv is used in training

## Pretrained model and Performance

[res101_lighthead_rcnn@ Google drive](https://drive.google.com/file/d/10Ku_G2FABjtEjWp3XWVmuguPkpbaTGV4/view?usp=sharing)

|                      Implementation                      | mAP@0.5:0.95/all | mAP@0.5/all | mAP@0.75/all | mAP:0.5:0.95/small | mAP:0.5:0.95/medium | mAP:0.5:0.95/large |
| :------------------------------------------------------: | :--------------: | :---------: | :----------: | :----------------: | :-----------------: | :----------------: |
| [Original](https://github.com/zengarden/light_head_rcnn) |      0.400       |    0.621    |    0.429     |       0.225        |        0.446        |       0.540        |
|                           Ours                           |      0.3963      |    0.601    |    0.432     |       0.217        |        0.447        |       0.539        |

```python
map/iou=0.50:0.95/area=all/max_dets=100: 0.396344
map/iou=0.50/area=all/max_dets=100: 0.601353
map/iou=0.75/area=all/max_dets=100: 0.432155
map/iou=0.50:0.95/area=small/max_dets=100: 0.217397
map/iou=0.50:0.95/area=medium/max_dets=100: 0.447772
map/iou=0.50:0.95/area=large/max_dets=100: 0.538968
mar/iou=0.50:0.95/area=all/max_dets=1: 0.319597
mar/iou=0.50:0.95/area=all/max_dets=10: 0.496089
mar/iou=0.50:0.95/area=all/max_dets=100: 0.513308
mar/iou=0.50:0.95/area=small/max_dets=100: 0.296478
mar/iou=0.50:0.95/area=medium/max_dets=100: 0.578120
mar/iou=0.50:0.95/area=large/max_dets=100: 0.692352
```

I have only trained it with 18 epochs instead of 30 epochs in the original paper due to it took too long time on my single 1080TI card system.

## How to use

### Preparation

1. #### Prepare COCO dataset.

   It is recommended to symlink the dataset root to `$Lighthead-RCNN-Pytorch/data`.

2. #### Create following folders and download the [pretrained model](https://drive.google.com/file/d/10Ku_G2FABjtEjWp3XWVmuguPkpbaTGV4/view?usp=sharing) to work_space/final

```shell
Lighthead-RCNN-Pytorch
├── functions
├── models
├── utils
├── data
│   ├── coco2014
│   │   ├── annotations
│   │   ├── train2014
│   │   ├── val2014
├── work_space
│   ├── model
│   ├── log
│   ├── final
│   ├── save
```

### Installation 

1. Install PyTorch 0.4.1 and torchvision following the [official instructions](https://pytorch.org/).

2. Clone the this repository.

   ```shell
   git clone https://github.com/TreB1eN/Lighthead-RCNN-in-Pytorch0.4.1.git
   cd Lighthead-RCNN-in-Pytorch0.4.1/
   ```

3. Install dependencies

   ```shell
   pip install -r requirements.txt
   ```

4. Compile cuda extensions.

   ```shell
   ./compile.sh  # or "PYTHON=python3 ./compile.sh" if you use system python3 without virtual environments
   ```

### Inference

```shell
python detect.py -f data/person.jpg -o data/person_detected.jpg
```

### Evaluate

```shell
python eval.py
```

### Train

```shell
python train.py 
```

------

More detailed configuration is in [utils.config.py](https://github.com/TreB1eN/Lighthead-RCNN-in-Pytorch0.4.1/blob/master/utils/config.py)

## References

[chainer-light-head-rcnn](https://github.com/knorth55/chainer-light-head-rcnn)

[light_head_rcnn](https://github.com/zengarden/light_head_rcnn)

[paper](https://arxiv.org/abs/1711.07264) 

## Demos

![](https://github.com/TreB1eN/Lighthead-RCNN-in-Pytorch0.4.1/blob/master/data/city_detected.jpg)

![](https://github.com/TreB1eN/Lighthead-RCNN-in-Pytorch0.4.1/blob/master/data/dinner_detected.jpg)

![](https://github.com/TreB1eN/Lighthead-RCNN-in-Pytorch0.4.1/blob/master/data/football_detected.jpg)

![](https://github.com/TreB1eN/Lighthead-RCNN-in-Pytorch0.4.1/blob/master/data/person_detected.jpg)