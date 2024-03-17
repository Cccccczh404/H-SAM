# H-SAM

## Paper

<b>Unleashing the Potential of SAM for Medical Adaptation via Hierarchical Decoding</b> <br/>
[Zhiheng Cheng](https://scholar.google.com/citations?hl=zh-CN&user=JUy6POQAAAAJ), [Qingyue Wei](https://profiles.stanford.edu/qingyue-wei), [Hongru Zhu](https://pages.jh.edu/hzhu38/), [Yan Wang](https://wangyan921.github.io/), [Wei Shao](https://swsamleo.github.io/wei_shao.github.io/), and [Yuyin Zhou](https://yuyinzhou.github.io/) <br/>
CVPR 2024 <br/>
[paper] | [code](https://github.com/Cccccczh404/H-SAM)

## 0. Installation

```bash
git clone https://github.com/Cccccczh404/H-SAM.git
```
Please run the following commands to create an environment and obtain requirements.
```bash
conda create -n H-SAM python=3.10
conda activate H-SAM
pip install -r requirements.txt
```

## 1. Prepare your datasets and pretained model
#### 1.1 Please download the processed [training set](https://drive.google.com/file/d/1zuOQRyfo0QYgjcU_uZs0X3LdCnAC2m3G/view?usp=share_link), whose resolution is `224x224`, and put it in `<Your folder>`. Then, unzip and delete this file. We also prepare the [training set](https://drive.google.com/file/d/1F42WMa80UpH98Pw95oAzYDmxAAO2ApYg/view?usp=share_link) with resolution `512x512` for reference, the `224x224` version of training set is downsampled from the `512x512` version.
#### 1.2 Please download the [testset](https://drive.google.com/file/d/1RczbNSB37OzPseKJZ1tDxa5OO1IIICzK/view?usp=share_link) and put it in the ./testset folder. Then, unzip and delete this file.
#### 1.3 Please download the pretrained SAM models from the original SAM repository put them in the ./checkpoints folder. 
#### ViT-B: (https://drive.google.com/file/d/1RczbNSB37OzPseKJZ1tDxa5OO1IIICzK/view?usp=share_link) 
#### ViT-L: (https://drive.google.com/file/d/1RczbNSB37OzPseKJZ1tDxa5OO1IIICzK/view?usp=share_link) 

## 2. Training
Use the train.py file for training models. An example script is
```

```

## 3. Testing
Use the test.py file for testing models. An example script is
```

```

## Acknowledgement
We appreciate the developers of [Segment Anything Model](https://github.com/facebookresearch/segment-anything) and the provider of the [Synapse multi-organ segmentation dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789). Our code of H-SAM is built upon [SAMed](https://github.com/hitachinsk/SAMed), and we express our gratitude to these awesome projects.


