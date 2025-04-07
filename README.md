# VLN-MBA-VisualPerturbations

This repository is the official implementation of **[Seeing is Believing? Enhancing Vision-Language Navigation using Visual Perturbations](https://arxiv.org/abs/2409.05552).**,**(IJCNN2025 Accepted)** 

>Vision-and-language navigation (VLN) enables the agent to navigate to a remote location following the natural language instruction in 3D environments. To represent the previously visited environment, most approaches for VLN implement memory using recurrent states, topological maps, or top-down semantic maps. In contrast to these approaches, we build the top-down egocentric and dynamically growing Grid Memory Map (i.e., GridMM) to structure the visited environment. From a global perspective, historical observations are projected into a unified grid map in a top-down view, which can better represent the spatial relations of the environment. From a local perspective, we further propose an instruction relevance aggregation method to capture fine-grained visual clues in each grid region. Extensive experiments are conducted on both the REVERIE, R2R, SOON datasets in the discrete environments, and the R2R-CE dataset in the continuous environments, showing the superiority of our proposed method.

> ![image](https://github.com/user-attachments/assets/c4d1ab8f-bfaf-4c3f-8198-0f134254e32a)


## 1. Requirements

1. Install Matterport3D simulator for `R2R`, `REVERIE` and `SOON`: follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator).
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```

2. Install requirements:
```setup
conda create --name MBA python=3.8.5
conda activate MBA
pip install -r requirements.txt
```

## 2. Data Download

1. Download data from [Dropbox](https://www.dropbox.com/sh/u3lhng7t2gq36td/AABAIdFnJxhhCg2ItpAhMtUBa?dl=0), including processed annotations, features and pretrained models of REVERIE, SOON, R2R and R4R datasets. Put the data in `datasets' directory.

2. Download pretrained lxmert
```
mkdir -p datasets/pretrained 
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P datasets/pretrained
```
3. Download Clip-based rgb feature and Depth feature (glbson and imagenet) form (链接: [https://pan.baidu.com/s/1lKend8xnwuy1uxn-aIDBtw?pwd=n8gv](https://pan.baidu.com/s/1lKend8xnwuy1uxn-aIDBtw?pwd=n8gv) 提取码: n8gv)
The ground truth depth image (undistorted_depth_images) is obtained from the [Matterport Simulator](https://github.com/peteanderson80/Matterport3DSimulator), and depth view features are extracted through here:
```
python get_depth.py
```
 The code is referenced from [HAMT](https://github.com/cshizhe/VLN-HAMT/tree/main/preprocess) and [here](https://github.com/zehao-wang/LAD/tree/main/preprocess)

## 3. Pretraining

The pretrained ckpts for REVERIE, R2R, SOON  is at [here](https://pan.baidu.com/s/1lKend8xnwuy1uxn-aIDBtw?pwd=n8gv). You can also pretrain the model by yourself, just change the pre training RGB of Duet from vit based to clip based. 
Combine behavior cloning and auxiliary proxy tasks in pretraining:
```pretrain
cd pretrain_src
bash run_r2r.sh # (run_reverie.sh, run_soon.sh)
```

## 4. Fine-tuning & Evaluation for `R2R`, `REVERIE` and `SOON`

Use pseudo interative demonstrator to fine-tune the model:
```finetune
cd map_nav_src
bash scripts/run_r2r.sh # (run_reverie.sh, run_soon.sh)
```

## 5. Test
Our report results on the test set are from the official website of Eval.ai. 
R2R: [https://eval.ai/web/challenges/challenge-page/97/submission](https://eval.ai/web/challenges/challenge-page/97/submission)
REVERIE: [https://eval.ai/web/challenges/challenge-page/606/overview](https://eval.ai/web/challenges/challenge-page/606/overview)
SOON: [https://eval.ai/web/challenges/challenge-page/1275/overview](https://eval.ai/web/challenges/challenge-page/1275/overview)
![image](https://github.com/user-attachments/assets/6a245c0a-8c61-4500-b3b8-26b269bce4a5)


## 6. Additional Resources 

1) Panoramic trajectory visualization is provided by [Speaker-Follower](https://gist.github.com/ronghanghu/d250f3a997135c667b114674fc12edae).
2) Top-down maps for Matterport3D are available in [NRNS](https://github.com/meera1hahn/NRNS).
3) Instructions for extracting image features from Matterport3D scenes can be found in [VLN-HAMT](https://github.com/cshizhe/VLN-HAMT).

(copy from [goat](https://github.com/CrystalSixone/VLN-GOAT)) We extend our gratitude to all the authors for their significant contributions and for sharing their resources.



## 7. Citation

```bibtex
@article{zhang2024seeing,
  title={Seeing is Believing? Enhancing Vision-Language Navigation using Visual Perturbations},
  author={Zhang, Xuesong and Li, Jia and Xu, Yunbo and Hu, Zhenzhen and Hong, Richang},
  journal={arXiv preprint arXiv:2409.05552},
  year={2024}
}
  ```

## Acknowledgments
Our code is based on [VLN-DUET](https://github.com/cshizhe/VLN-DUET) and partially referenced from [HAMT](https://github.com/cshizhe/VLN-HAMT/tree/main/preprocess) for extract view features. Thanks for their great works!

