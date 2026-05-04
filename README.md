# DoRA - Fine-tuned for Nature Using Egocentric GoPro Videos

This repository is a clone of [DoRA](https://shashankvkt.github.io/dora) (ICLR 2024), from the paper *"Is ImageNet worth 1 video? Learning strong image encoders from 1 long unlabelled video"* by Venkataramanan et al.

Original paper: [[`arXiv`](https://arxiv.org/abs/2310.08584)] [[`paper`](https://openreview.net/forum?id=Yen1lGns2o)] [[`dataset`](https://uvaauas.figshare.com/articles/dataset/Dora_WalkingTours_Dataset_ICLR_2024_/25189275)] [[`Project Page`](https://shashankvkt.github.io/dora)]

## Overview

DoRA (DiscOvering and tRAcking) is a self-supervised learning method that leverages the rich information in long, unlabelled videos. It uses attention from the [CLS] token of distinct heads in a vision transformer to identify and consistently track multiple objects within frames across temporal sequences, applying a teacher-student distillation loss on them, without any off-the-shelf object tracker or optical flow.

## Presentation

[Project Presentation (Google Slides)](https://docs.google.com/presentation/d/1zjc6SHeHqZQelLO62faekV2b5ywO9w5YlL5cMp5XKQo/edit?usp=sharing)

## What we did

The original DoRA model was pretrained on urban-oriented, egocentric WalkingTours videos (residential areas, parks, markets, waterfronts). We investigated whether fine-tuning on **nature-oriented content** would help the model learn better representations of natural scenes and environments.

### Our dataset

We captured our own videos using a GoPro Hero10 Black -- multiple egocentric, 4K/5K 30fps clips (seconds to minutes long each) featuring nature scenes: sea, mountains, trees, animals, and combinations thereof.

### Approach

We fine-tuned the DoRA checkpoint (trained on WalkingTours) on our GoPro nature videos for 10 additional epochs (~3 hours). We also modified the preprocessing pipeline: instead of applying random transformations on-the-fly during each epoch (the original approach), we preprocessed the entire dataset once beforehand. This saved significant time (~15 hours total one-time cost), at the expense of losing per-epoch stochasticity in transformations.

### Evaluation

To measure whether the model's understanding of nature improved, we performed an extrinsic evaluation: we froze the transformer weights and trained a classifier on top using Linear Probing and KNN on the [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (sea, mountains, glaciers, streets, buildings, trees; 6 classes).

### Results
- The finetuned model achieved higher accuracy on classes whose concepts were present during nature-video fine-tuning (sea, mountains, trees, buildings, streets)
- Lower accuracy only on the 'Glacier' class, which was not represented in our fine-tuning data
- The results suggest that with just 10 additional epochs of targeted fine-tuning, the model learned improved representations of nature-related concepts

## Prerequisites
conda activate dora

conda install pytz

conda install colorlog

conda install h5py


## Fine-tuning


python3 main.py --arch vit_small  \
--output_dir /home/giorgos/computer_vision/project/computer_vision_project/checkpoint --optimizer adamw  \
--use_bn_in_head False --out_dim 65536 --batch_size_per_gpu 12 --local_crops_number 2 --epochs 110  \
--num_workers 20 --lr 0.0001 --min_lr 0.00001 --norm_last_layer False --warmup_teacher_temp_epochs 1 --weight_decay 0.04 --weight_decay_end 0.4  \
--frame_per_clip 8 --step_between_clips 20  --warmup_epochs 1 --from_start False

 -------------------------------------------------------------------------
## Linear Probing

python eval_linear.py \
--batch_size_per_gpu 512 --n_last_blocks 4 --avgpool_patchtokens 0 --arch vit_small --lr 0.002  \
--pretrained_weights /home/giorgos/computer_vision/project/computer_vision_project/checkpoint/checkpoint.pth \
--output_dir /home/giorgos/computer_vision/project/computer_vision_project/output_intel/ --epochs 10 --dataset_name intel \
--num_workers 20 --num_labels 6

## KNN
python3 eval_knn.py --arch vit_small --checkpoint_key teacher --pretrained_weights /home/giorgos/computer_vision/project/computer_vision_project/checkpoint/checkpoint_DoRA_100_WT-all.pth

python3 eval_knn.py --arch vit_small --checkpoint_key teacher --pretrained_weights /home/giorgos/computer_vision/project/computer_vision_project/checkpoint/checkpoint.pth
