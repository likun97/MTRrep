# MTRre



## Some re-learnings&&annotations about [Motion Transformer (MTR): A Strong Baseline for Multimodal Motion Prediction in Autonomous Driving](https://github.com/sshaoshuai/MTR).


## Env && Installation

```sh
# basic env: torch1.1+cuda11.1
conda activate open-mmlab
pip install -r requirements.txt
python setup.py develop # to generate ./build/ & ./MotionTransformer.egg-info/
# knn_cuda_xx.so and attention_cuda_xx.so

```

## Dataset preparation

**Step 1:** Download the [Waymo Open Motion Dataset](https://waymo.com/open/download/) provided with [google cloud](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_2_0?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false): 


**Step 2:** Install Google Cloud SDK 
[install docs](https://cloud.google.com/sdk/docs/proxy-settings?hl=zh-cn)
[install setting](https://cloud.google.com/sdk/docs/install?hl=zh-cn#deb)

**gloud init**

**Step 3:** Install the [Waymo Open Dataset API](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md) as follows: 
```sh
pip install waymo-open-dataset-tf-2-6-0
```


**Step 4:** Preprocess the dataset:
```sh
# python data_preprocess.py ../../../data/waymo/scenario/  ../../../data/waymo
cd ./mtrcore/datasets/waymo
python data_preprocess.py \
    /mnt/nvme1n1/data/waymo_open_dataset_motion_v_1_2_0/scenario/  \
    /mnt/nvme1n1/data/waymo_open_dataset_motion_v_1_2_0
```

<!-- 
raise ValueError Occurs while "def decode_map_features_from_proto(map_features)"
https://github.com/sshaoshuai/MTR/issues/8, 11, 27, 60
-->


## Training instruction
For example, train with 8 GPUs: 
``` sh
cd tools

# bash scripts/dist_train.sh 8 --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size 80 --epochs 30 --extra_tag my_first_exp

# https://github.com/sshaoshuai/MTR/issues/52
# python train.py --cfg_file cfgs/waymo/mtr+100_percent_data_20s.yaml --batch_size 10 --epochs 30 --extra_tag my_first_exp



source env.sh
python tools/train.py --cfg_file tools/train_cfgs/waymo/mtr+100_percent_data.yaml --batch_size 2 --epochs 500 --extra_tag my_exp_0328


```