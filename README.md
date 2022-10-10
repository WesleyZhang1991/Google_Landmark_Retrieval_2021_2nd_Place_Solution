# Google_Landmark_Retrieval_2021_2nd_Place_Solution
The 2nd place solution of 2021 google landmark retrieval on kaggle. 

## Environment

We use cuda 11.1/python 3.7/torch 1.9.1/torchvision 0.8.1 for training and testing.

Download imagenet pretrained model ResNeXt101ibn and SEResNet101ibn from [IBN-Net](https://github.com/XingangPan/IBN-Net). ResNest101 and ResNeSt269 can be found in [ResNest](https://github.com/zhanghang1989/ResNeSt). 

## Prepare data

1. Download GLDv2 full version from the [official site](https://github.com/cvdfoundation/google-landmark).

2. Run `python tools/generate_gld_list.py`. This will generate `clean`, `c2x`, `trainfull` and `all` data for different stage of training.

3. Validation annotation comes from all 1129 images in GLDv2. We expand the competition index set to [index_expand](https://drive.google.com/file/d/116L3o3twAo18IYfZNoD2vtHp3FfyvKzM/view?usp=sharing). Each query could find all its GTs in the expanded index set and the validation could be more accurate.


## Train

We use 8 GPU (32GB/16GB) for training. The evaluation metric in landmark retrieval is different from person re-identification. Due to the validation scale, we skip the validation stage during training and just use the model from last epoch for evaluation.

### Fast Train Script

To make quick experiments, we provide scripts for `R50_256` trained for `clean` subset. This setting trains very fast and is helpful for debug.
```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --master_port 55555 --max_restarts 0 train.py --config_file configs/GLDv2/R50_256.yml
```

### Whole Train Pipeline

The whole training pipeline for SER101ibn backbone is listed below. Other backbones and input size can be modified accordingly.

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --master_port 55555 --max_restarts 0 train.py --config_file configs/GLDv2/SER101ibn_384.yml
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --master_port 55555 --max_restarts 0 train.py --config_file configs/GLDv2/SER101ibn_384_finetune.yml
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --master_port 55555 --max_restarts 0 train.py --config_file configs/GLDv2/SER101ibn_512_finetune.yml
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --master_port 55555 --max_restarts 0 train.py --config_file configs/GLDv2/SER101ibn_512_all.yml
```

## Inference(notebooks)

* With four models trained, cd to `submission/code/` and modify `settings` in `landmark_retrieval.py` properly.

* Then run `eval_retrieval.sh` to get submission file and evaluate on validation set offline.


### General Settings
```
REID_EXTRACT_FLAG: Skip feature extraction when using offline code.
FEAT_DIR: Save cached features.
IMAGE_DIR: competition image dir. We make a soft link for competition data at submission/input/landmark-retrieval-2021/
RAW_IMAGE_DIR: origin GLDv2 dir
MODEL_DIR: the latest models for submission
META_DIR: saves meta files for rerank purpose
LOCAL_MATCHING and KR_FLAG disabled for our submission.
```

### Fast Inference Script

Use `R50_256` model trained from `clean` subset correspongding to the fast train script. Set `CATEGORY_RERANK` and `REF_SET_EXTRACT` to False. You will get about mAP=32.84% for the validation set.


### Whole Inference Pipeline

* Copy `cache_all_list.pkl`, `cache_index_train_list.pkl` and `cache_full_list.pkl` from `cache` to `submission/input/meta-data-final`

* Set `REF_SET_EXTRACT` to `True` to extract features for `all` images of GLDv2. This will save about 4.9 million 512 dim features for each model in `submission/input/meta-data-final`.

* Set `REF_SET_EXTRACT` to `False` and `CATEGORY_RERANK` to `before_merge`. This will load the precomputed features and run the proposed Landmark-Country aware rerank.

* The notebooks on kaggle is exactly the same file as in `base_landmark.py` and `landmark_retrieval.py`. We also upload the same notebooks as in kaggle in `kaggle.ipynb`.


## Kaggle and ICCV workshops

* The challenge is held on [kaggle](https://www.kaggle.com/c/landmark-retrieval-2021) and the leaderboard can be found [here](https://www.kaggle.com/c/landmark-retrieval-2021/leaderboard). We rank 2nd(2/263) in this challenge.

* Kaggle Discussion post link [here](https://www.kaggle.com/c/landmark-retrieval-2021/discussion/277273)

* ICCV workshop [slides](https://github.com/WesleyZhang1991/Google_Landmark_Retrieval_2021_2nd_Place_Solution/blob/master/ILR21_RET_2nd-slides.pdf) and [videos](https://www.youtube.com/watch?v=bkT2Judxf_s).


## Thanks
The code is motivated by [AICITY2021_Track2_DMT](https://github.com/michuanhaohao/AICITY2021_Track2_DMT), [2020_1st_recognition_solution](https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place), [2020_2nd_recognition_solution](https://github.com/bestfitting/instance_level_recognition), [2020_1st_retrieval_solution](https://github.com/seungkee/google_landmark_retrieval_2020_1st_place_solution).


## Citation

If you find our work useful in your research, please consider citing:
```
@inproceedings{zhang2021landmark,
 title={2nd Place Solution to Google Landmark Retrieval 2021},
 author={Zhang, Yuqi and Xu, Xianzhe and Chen, Weihua and Wang, Yaohua and Zhang, Fangyi and Wang Fan and Li Hao},
 journal={arXiv preprint arXiv:2110.04294},
 year={2021}
}
```
