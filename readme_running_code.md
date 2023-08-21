## 1. Dataset Preparation
Dataset preparation的过程见`PoseFormer/readme_running_code.md`
`prepare_data_h36m.py`脚本的comments见`PoseFormer/data/`

将预处理完的data放值该目录下的`data`目录下。

## 2. Download pre-trained model
The pretrained models can be downloaded from AWS. Put pretrained_h36m_cpn.bin (for Human3.6M) and/or pretrained_humaneva15_detectron.bin (for HumanEva) in the checkpoint/ directory (create it if it does not exist).
```commandline
mkdir checkpoint
cd checkpoint
wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_cpn.bin
wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_humaneva15_detectron.bin
cd ..
```

## 3. Test pre-trained model
Note: `Documentation.md`中有输入参数的具体解释，

To test on Human3.6M, run:
- CPN detected 2D pose as input

    Note: we need to download 2D detection, as shown in https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md, and place the `.npz` file in data directory

  ```commandline
  python run.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin
  ```
- 2D pose gt as input
    ```commandline
      python run.py -k gt -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin
    ```  


## 4. Run on custom videos
see in `Inference.md`: 包括下载2D pose ddetector预测2d kps，将2d kps整合为`.npz`的数据形式，再使用预训练模型获得视频的结果。

## 5. Training
For Human3.6M:
- CPN detected 2D pose as input
    ```commandline
    python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3
    ```
- 2D pose gt as input
    ```commandline
    python run.py -e 80 -k gt -arc 3,3,3,3,3
    ```

By default, the application runs in training mode. This will train a new model for 80 epochs, using fine-tuned CPN detections. Expect a training time of 24 hours on a high-end Pascal GPU. If you feel that this is too much, or your GPU is not powerful enough, you can train a model with a smaller receptive field, e.g.
- arc 3,3,3,3 (81 frames) should require 11 hours and achieve 47.7 mm.
- arc 3,3,3 (27 frames) should require 6 hours and achieve 48.8 mm.

You could also lower the number of epochs from 80 to 60 with a negligible impact on the result.

## Semi-supervised training
To perform semi-supervised training, you just need to add the --subjects-unlabeled argument. In the example below, we use ground-truth 2D poses as input, and train supervised on just 10% of Subject 1 (specified by --subset 0.1). The remaining subjects are treated as unlabeled data and are used for semi-supervision.
```commandline
python run.py -k gt --subjects-train S1 --subset 0.1 --subjects-unlabeled S5,S6,S7,S8 -e 200 -lrd 0.98 -arc 3,3,3 --warmup 5 -b 64
```
This should give you an error around 65.2 mm. By contrast, if we only train supervised
```commandline
python run.py -k gt --subjects-train S1 --subset 0.1 -e 200 -lrd 0.98 -arc 3,3,3 -b 64
```
we get around 80.7 mm, which is significantly higher.

## Visualization
If you have the original Human3.6M videos, you can generate nice visualizations of the model predictions. For instance:
```commandline
python run.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin --render --viz-subject S11 --viz-action Walking --viz-camera 0 --viz-video "/path/to/videos/S11/Videos/Walking.54138969.mp4" --viz-output output.gif --viz-size 3 --viz-downsample 2 --viz-limit 60
```
The script can also export MP4 videos, and supports a variety of parameters (e.g. downsampling/FPS, size, bitrate). See DOCUMENTATION.md for more details.