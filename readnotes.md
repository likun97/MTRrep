
(open-mmlab) likun@likun-Default-string:~/WS/code/MTRre$ python tools/train.py --cfg_file tools/train_cfgs/waymo/mtr+100_percent_data.yaml --batch_size 2 --epochs 500 --extra_tag my_exp_03171
cfg.ROOT_DIR : /home/likun/WS/code/MTRre
2024-03-17 14:47:50,087   INFO  **********************Start logging**********************
2024-03-17 14:47:50,087   INFO  CUDA_VISIBLE_DEVICES=ALL
2024-03-17 14:47:50,087   INFO  cfg_file         tools/train_cfgs/waymo/mtr+100_percent_data.yaml
2024-03-17 14:47:50,087   INFO  batch_size       2
2024-03-17 14:47:50,087   INFO  epochs           500
2024-03-17 14:47:50,087   INFO  workers          8
2024-03-17 14:47:50,087   INFO  extra_tag        my_exp_03171
2024-03-17 14:47:50,087   INFO  ckpt             None
2024-03-17 14:47:50,087   INFO  pretrained_model None
2024-03-17 14:47:50,087   INFO  launcher         none
2024-03-17 14:47:50,087   INFO  tcp_port         18888
2024-03-17 14:47:50,087   INFO  without_sync_bn  True
2024-03-17 14:47:50,088   INFO  fix_random_seed  False
2024-03-17 14:47:50,088   INFO  ckpt_save_interval 2
2024-03-17 14:47:50,088   INFO  local_rank       None
2024-03-17 14:47:50,088   INFO  max_ckpt_save_num 5
2024-03-17 14:47:50,088   INFO  merge_all_iters_to_one_epoch False
2024-03-17 14:47:50,088   INFO  set_cfgs         None
2024-03-17 14:47:50,088   INFO  max_waiting_mins 0
2024-03-17 14:47:50,088   INFO  start_epoch      0
2024-03-17 14:47:50,088   INFO  save_to_file     False
2024-03-17 14:47:50,088   INFO  not_eval_with_train False
2024-03-17 14:47:50,088   INFO  logger_iter_interval 50
2024-03-17 14:47:50,088   INFO  ckpt_save_time_interval 300
2024-03-17 14:47:50,088   INFO  add_worker_init_fn False
2024-03-17 14:47:50,088   INFO  cfg.ROOT_DIR: /home/likun/WS/code/MTRre
2024-03-17 14:47:50,088   INFO  cfg.LOCAL_RANK: 0
2024-03-17 14:47:50,088   INFO  
cfg.DATA_CONFIG = edict()
2024-03-17 14:47:50,088   INFO  cfg.DATA_CONFIG.DATASET: WaymoDataset
2024-03-17 14:47:50,088   INFO  cfg.DATA_CONFIG.OBJECT_TYPE: ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']
2024-03-17 14:47:50,088   INFO  cfg.DATA_CONFIG.DATA_ROOT: /mnt/nvme1n1/data/waymo_open_dataset_motion_v_1_2_0
2024-03-17 14:47:50,088   INFO  
cfg.DATA_CONFIG.SPLIT_DIR = edict()
2024-03-17 14:47:50,088   INFO  cfg.DATA_CONFIG.SPLIT_DIR.train: processed_scenarios_training
2024-03-17 14:47:50,088   INFO  cfg.DATA_CONFIG.SPLIT_DIR.test: processed_scenarios_validation
2024-03-17 14:47:50,088   INFO  
cfg.DATA_CONFIG.INFO_FILE = edict()
2024-03-17 14:47:50,088   INFO  cfg.DATA_CONFIG.INFO_FILE.train: processed_scenarios_training_infos.pkl
2024-03-17 14:47:50,088   INFO  cfg.DATA_CONFIG.INFO_FILE.test: processed_scenarios_val_infos.pkl
2024-03-17 14:47:50,088   INFO  
cfg.DATA_CONFIG.SAMPLE_INTERVAL = edict()
2024-03-17 14:47:50,088   INFO  cfg.DATA_CONFIG.SAMPLE_INTERVAL.train: 1
2024-03-17 14:47:50,088   INFO  cfg.DATA_CONFIG.SAMPLE_INTERVAL.test: 1
2024-03-17 14:47:50,088   INFO  
cfg.DATA_CONFIG.INFO_FILTER_DICT = edict()
2024-03-17 14:47:50,088   INFO  cfg.DATA_CONFIG.INFO_FILTER_DICT.filter_info_by_object_type: ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']
2024-03-17 14:47:50,088   INFO  cfg.DATA_CONFIG.POINT_SAMPLED_INTERVAL: 1
2024-03-17 14:47:50,088   INFO  cfg.DATA_CONFIG.NUM_POINTS_EACH_POLYLINE: 20
2024-03-17 14:47:50,088   INFO  cfg.DATA_CONFIG.VECTOR_BREAK_DIST_THRESH: 1.0
2024-03-17 14:47:50,088   INFO  cfg.DATA_CONFIG.NUM_OF_SRC_POLYLINES: 768
2024-03-17 14:47:50,088   INFO  cfg.DATA_CONFIG.CENTER_OFFSET_OF_MAP: [30.0, 0]
2024-03-17 14:47:50,088   INFO  
cfg.MODEL = edict()
2024-03-17 14:47:50,088   INFO  
cfg.MODEL.CONTEXT_ENCODER = edict()
2024-03-17 14:47:50,088   INFO  cfg.MODEL.CONTEXT_ENCODER.NAME: MTREncoder
2024-03-17 14:47:50,088   INFO  cfg.MODEL.CONTEXT_ENCODER.NUM_OF_ATTN_NEIGHBORS: 16
2024-03-17 14:47:50,088   INFO  cfg.MODEL.CONTEXT_ENCODER.NUM_INPUT_ATTR_AGENT: 29
2024-03-17 14:47:50,088   INFO  cfg.MODEL.CONTEXT_ENCODER.NUM_INPUT_ATTR_MAP: 9
2024-03-17 14:47:50,088   INFO  cfg.MODEL.CONTEXT_ENCODER.NUM_CHANNEL_IN_MLP_AGENT: 256
2024-03-17 14:47:50,088   INFO  cfg.MODEL.CONTEXT_ENCODER.NUM_CHANNEL_IN_MLP_MAP: 64
2024-03-17 14:47:50,088   INFO  cfg.MODEL.CONTEXT_ENCODER.NUM_LAYER_IN_MLP_AGENT: 3
2024-03-17 14:47:50,088   INFO  cfg.MODEL.CONTEXT_ENCODER.NUM_LAYER_IN_MLP_MAP: 5
2024-03-17 14:47:50,088   INFO  cfg.MODEL.CONTEXT_ENCODER.NUM_LAYER_IN_PRE_MLP_MAP: 3
2024-03-17 14:47:50,088   INFO  cfg.MODEL.CONTEXT_ENCODER.D_MODEL: 256
2024-03-17 14:47:50,088   INFO  cfg.MODEL.CONTEXT_ENCODER.NUM_ATTN_LAYERS: 6
2024-03-17 14:47:50,088   INFO  cfg.MODEL.CONTEXT_ENCODER.NUM_ATTN_HEAD: 8
2024-03-17 14:47:50,088   INFO  cfg.MODEL.CONTEXT_ENCODER.DROPOUT_OF_ATTN: 0.1
2024-03-17 14:47:50,088   INFO  cfg.MODEL.CONTEXT_ENCODER.USE_LOCAL_ATTN: True
2024-03-17 14:47:50,088   INFO  
cfg.MODEL.MOTION_DECODER = edict()
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.NAME: MTRDecoder
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.OBJECT_TYPE: ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.CENTER_OFFSET_OF_MAP: [30.0, 0]
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.NUM_FUTURE_FRAMES: 80
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.NUM_MOTION_MODES: 6
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.INTENTION_POINTS_FILE: tools/IntentionPointsData/waymo/cluster_64_center_dict.pkl
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.D_MODEL: 512
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.NUM_DECODER_LAYERS: 6
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.NUM_ATTN_HEAD: 8
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.MAP_D_MODEL: 256
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.DROPOUT_OF_ATTN: 0.1
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.NUM_BASE_MAP_POLYLINES: 256
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.NUM_WAYPOINT_MAP_POLYLINES: 128
2024-03-17 14:47:50,088   INFO  
cfg.MODEL.MOTION_DECODER.LOSS_WEIGHTS = edict()
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.LOSS_WEIGHTS.cls: 1.0
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.LOSS_WEIGHTS.reg: 1.0
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.LOSS_WEIGHTS.vel: 0.5
2024-03-17 14:47:50,088   INFO  cfg.MODEL.MOTION_DECODER.NMS_DIST_THRESH: 2.5
2024-03-17 14:47:50,088   INFO  
cfg.OPTIMIZATION = edict()
2024-03-17 14:47:50,088   INFO  cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU: 10
2024-03-17 14:47:50,089   INFO  cfg.OPTIMIZATION.NUM_EPOCHS: 30
2024-03-17 14:47:50,089   INFO  cfg.OPTIMIZATION.OPTIMIZER: AdamW
2024-03-17 14:47:50,089   INFO  cfg.OPTIMIZATION.LR: 0.0001
2024-03-17 14:47:50,089   INFO  cfg.OPTIMIZATION.WEIGHT_DECAY: 0.01
2024-03-17 14:47:50,089   INFO  cfg.OPTIMIZATION.SCHEDULER: lambdaLR
2024-03-17 14:47:50,089   INFO  cfg.OPTIMIZATION.DECAY_STEP_LIST: [22, 24, 26, 28]
2024-03-17 14:47:50,089   INFO  cfg.OPTIMIZATION.LR_DECAY: 0.5
2024-03-17 14:47:50,089   INFO  cfg.OPTIMIZATION.LR_CLIP: 1e-06
2024-03-17 14:47:50,089   INFO  cfg.OPTIMIZATION.GRAD_NORM_CLIP: 1000.0
2024-03-17 14:47:50,089   INFO  cfg.TAG: mtr+100_percent_data
2024-03-17 14:47:50,089   INFO  cfg.EXP_GROUP_PATH: train_cfgs/waymo
2024-03-17 14:47:50,091   INFO  Start to load infos from /mnt/nvme1n1/data/waymo_open_dataset_motion_v_1_2_0/processed_scenarios_training_infos.pkl
2024-03-17 14:47:54,301   INFO  Total scenes before filters: 486995
2024-03-17 14:47:57,910   INFO  Total scenes after filter_info_by_object_type: 486995
2024-03-17 14:47:57,923   INFO  Total scenes after filters: 486995
2024-03-17 14:48:01,575   INFO  MotionTransformer(
  (context_encoder): MTREncoder(
    (agent_polyline_encoder): PointNetPolylineEncoder(
      (pre_mlps): Sequential(
        (0): Linear(in_features=30, out_features=256, bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (mlps): Sequential(
        (0): Linear(in_features=512, out_features=256, bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=256, out_features=256, bias=False)
        (4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
      )
      (out_mlps): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
      )
    )
    (map_polyline_encoder): PointNetPolylineEncoder(
      (pre_mlps): Sequential(
        (0): Linear(in_features=9, out_features=64, bias=False)
        (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=64, out_features=64, bias=False)
        (4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=64, out_features=64, bias=False)
        (7): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU()
      )
      (mlps): Sequential(
        (0): Linear(in_features=128, out_features=64, bias=False)
        (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=64, out_features=64, bias=False)
        (4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
      )
      (out_mlps): Sequential(
        (0): Linear(in_features=64, out_features=64, bias=True)
        (1): ReLU()
        (2): Linear(in_features=64, out_features=256, bias=True)
      )
    )
    (self_attn_layers): ModuleList(
      (0): TransformerEncoderLayer(
        (self_attn): MultiheadAttentionLocal(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=1024, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=1024, out_features=256, bias=True)
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
      (1): TransformerEncoderLayer(
        (self_attn): MultiheadAttentionLocal(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=1024, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=1024, out_features=256, bias=True)
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
      (2): TransformerEncoderLayer(
        (self_attn): MultiheadAttentionLocal(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=1024, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=1024, out_features=256, bias=True)
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
      (3): TransformerEncoderLayer(
        (self_attn): MultiheadAttentionLocal(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=1024, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=1024, out_features=256, bias=True)
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
      (4): TransformerEncoderLayer(
        (self_attn): MultiheadAttentionLocal(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=1024, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=1024, out_features=256, bias=True)
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
      (5): TransformerEncoderLayer(
        (self_attn): MultiheadAttentionLocal(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=1024, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=1024, out_features=256, bias=True)
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (motion_decoder): MTRDecoder(
    (in_proj_center_obj): Sequential(
      (0): Linear(in_features=256, out_features=512, bias=True)
      (1): ReLU()
      (2): Linear(in_features=512, out_features=512, bias=True)
    )
    (in_proj_obj): Sequential(
      (0): Linear(in_features=256, out_features=512, bias=True)
      (1): ReLU()
      (2): Linear(in_features=512, out_features=512, bias=True)
    )
    (obj_decoder_layers): ModuleList(
      (0): TransformerDecoderLayer(
        (sa_qcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_qpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_kcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_kpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_v_proj): Linear(in_features=512, out_features=512, bias=True)
        (self_attn): MultiheadAttention(
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (ca_qcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_qpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_kcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_kpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_v_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_qpos_sine_proj): Linear(in_features=512, out_features=512, bias=True)
        (cross_attn): MultiheadAttention(
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=512, bias=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
      (1): TransformerDecoderLayer(
        (sa_qcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_qpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_kcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_kpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_v_proj): Linear(in_features=512, out_features=512, bias=True)
        (self_attn): MultiheadAttention(
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (ca_qcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_qpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_kcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_kpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_v_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_qpos_sine_proj): Linear(in_features=512, out_features=512, bias=True)
        (cross_attn): MultiheadAttention(
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=512, bias=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
      (2): TransformerDecoderLayer(
        (sa_qcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_qpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_kcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_kpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_v_proj): Linear(in_features=512, out_features=512, bias=True)
        (self_attn): MultiheadAttention(
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (ca_qcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_qpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_kcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_kpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_v_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_qpos_sine_proj): Linear(in_features=512, out_features=512, bias=True)
        (cross_attn): MultiheadAttention(
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=512, bias=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
      (3): TransformerDecoderLayer(
        (sa_qcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_qpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_kcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_kpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_v_proj): Linear(in_features=512, out_features=512, bias=True)
        (self_attn): MultiheadAttention(
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (ca_qcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_qpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_kcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_kpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_v_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_qpos_sine_proj): Linear(in_features=512, out_features=512, bias=True)
        (cross_attn): MultiheadAttention(
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=512, bias=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
      (4): TransformerDecoderLayer(
        (sa_qcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_qpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_kcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_kpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_v_proj): Linear(in_features=512, out_features=512, bias=True)
        (self_attn): MultiheadAttention(
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (ca_qcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_qpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_kcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_kpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_v_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_qpos_sine_proj): Linear(in_features=512, out_features=512, bias=True)
        (cross_attn): MultiheadAttention(
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=512, bias=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
      (5): TransformerDecoderLayer(
        (sa_qcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_qpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_kcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_kpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_v_proj): Linear(in_features=512, out_features=512, bias=True)
        (self_attn): MultiheadAttention(
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (ca_qcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_qpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_kcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_kpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_v_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_qpos_sine_proj): Linear(in_features=512, out_features=512, bias=True)
        (cross_attn): MultiheadAttention(
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=512, bias=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
    )
    (in_proj_map): Sequential(
      (0): Linear(in_features=256, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=256, bias=True)
    )
    (map_decoder_layers): ModuleList(
      (0): TransformerDecoderLayer(
        (sa_qcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_qpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_kcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_kpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_v_proj): Linear(in_features=256, out_features=256, bias=True)
        (self_attn): MultiheadAttention(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (ca_qcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_qpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_kcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_kpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_v_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_qpos_sine_proj): Linear(in_features=256, out_features=256, bias=True)
        (cross_attn): MultiheadAttentionLocal(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=1024, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=1024, out_features=256, bias=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
      (1): TransformerDecoderLayer(
        (sa_qcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_qpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_kcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_kpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_v_proj): Linear(in_features=256, out_features=256, bias=True)
        (self_attn): MultiheadAttention(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (ca_qcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_qpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_kcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_kpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_v_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_qpos_sine_proj): Linear(in_features=256, out_features=256, bias=True)
        (cross_attn): MultiheadAttentionLocal(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=1024, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=1024, out_features=256, bias=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
      (2): TransformerDecoderLayer(
        (sa_qcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_qpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_kcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_kpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_v_proj): Linear(in_features=256, out_features=256, bias=True)
        (self_attn): MultiheadAttention(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (ca_qcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_qpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_kcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_kpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_v_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_qpos_sine_proj): Linear(in_features=256, out_features=256, bias=True)
        (cross_attn): MultiheadAttentionLocal(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=1024, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=1024, out_features=256, bias=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
      (3): TransformerDecoderLayer(
        (sa_qcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_qpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_kcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_kpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_v_proj): Linear(in_features=256, out_features=256, bias=True)
        (self_attn): MultiheadAttention(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (ca_qcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_qpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_kcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_kpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_v_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_qpos_sine_proj): Linear(in_features=256, out_features=256, bias=True)
        (cross_attn): MultiheadAttentionLocal(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=1024, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=1024, out_features=256, bias=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
      (4): TransformerDecoderLayer(
        (sa_qcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_qpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_kcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_kpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_v_proj): Linear(in_features=256, out_features=256, bias=True)
        (self_attn): MultiheadAttention(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (ca_qcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_qpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_kcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_kpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_v_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_qpos_sine_proj): Linear(in_features=256, out_features=256, bias=True)
        (cross_attn): MultiheadAttentionLocal(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=1024, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=1024, out_features=256, bias=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
      (5): TransformerDecoderLayer(
        (sa_qcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_qpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_kcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_kpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_v_proj): Linear(in_features=256, out_features=256, bias=True)
        (self_attn): MultiheadAttention(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (ca_qcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_qpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_kcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_kpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_v_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_qpos_sine_proj): Linear(in_features=256, out_features=256, bias=True)
        (cross_attn): MultiheadAttentionLocal(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=1024, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=1024, out_features=256, bias=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
    )
    (map_query_content_mlps): ModuleList(
      (0): Linear(in_features=512, out_features=256, bias=True)
      (1): Linear(in_features=512, out_features=256, bias=True)
      (2): Linear(in_features=512, out_features=256, bias=True)
      (3): Linear(in_features=512, out_features=256, bias=True)
      (4): Linear(in_features=512, out_features=256, bias=True)
      (5): Linear(in_features=512, out_features=256, bias=True)
    )
    (map_query_embed_mlps): Linear(in_features=512, out_features=256, bias=True)
    (obj_pos_encoding_layer): Sequential(
      (0): Linear(in_features=2, out_features=512, bias=True)
      (1): ReLU()
      (2): Linear(in_features=512, out_features=512, bias=True)
      (3): ReLU()
      (4): Linear(in_features=512, out_features=512, bias=True)
    )
    (dense_future_head): Sequential(
      (0): Linear(in_features=1024, out_features=512, bias=False)
      (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Linear(in_features=512, out_features=512, bias=False)
      (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU()
      (6): Linear(in_features=512, out_features=560, bias=True)
    )
    (future_traj_mlps): Sequential(
      (0): Linear(in_features=320, out_features=512, bias=True)
      (1): ReLU()
      (2): Linear(in_features=512, out_features=512, bias=True)
      (3): ReLU()
      (4): Linear(in_features=512, out_features=512, bias=True)
    )
    (traj_fusion_mlps): Sequential(
      (0): Linear(in_features=1024, out_features=512, bias=True)
      (1): ReLU()
      (2): Linear(in_features=512, out_features=512, bias=True)
      (3): ReLU()
      (4): Linear(in_features=512, out_features=512, bias=True)
    )
    (intention_query_mlps): Sequential(
      (0): Linear(in_features=512, out_features=512, bias=False)
      (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Linear(in_features=512, out_features=512, bias=True)
    )
    (query_feature_fusion_layers): ModuleList(
      (0): Sequential(
        (0): Linear(in_features=1280, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=True)
      )
      (1): Sequential(
        (0): Linear(in_features=1280, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=True)
      )
      (2): Sequential(
        (0): Linear(in_features=1280, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=True)
      )
      (3): Sequential(
        (0): Linear(in_features=1280, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=True)
      )
      (4): Sequential(
        (0): Linear(in_features=1280, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=True)
      )
      (5): Sequential(
        (0): Linear(in_features=1280, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=True)
      )
    )
    (motion_reg_heads): ModuleList(
      (0): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=512, out_features=560, bias=True)
      )
      (1): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=512, out_features=560, bias=True)
      )
      (2): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=512, out_features=560, bias=True)
      )
      (3): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=512, out_features=560, bias=True)
      )
      (4): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=512, out_features=560, bias=True)
      )
      (5): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=512, out_features=560, bias=True)
      )
    )
    (motion_cls_heads): ModuleList(
      (0): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=512, out_features=1, bias=True)
      )
      (1): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=512, out_features=1, bias=True)
      )
      (2): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=512, out_features=1, bias=True)
      )
      (3): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=512, out_features=1, bias=True)
      )
      (4): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=512, out_features=1, bias=True)
      )
      (5): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=512, out_features=1, bias=True)
      )
    )
  )
)
2024-03-17 14:48:01,579   INFO  Total number of parameters: 65781334
2024-03-17 14:48:01,579   INFO  Start to load infos from /mnt/nvme1n1/data/waymo_open_dataset_motion_v_1_2_0/processed_scenarios_val_infos.pkl
2024-03-17 14:48:01,771   INFO  Total scenes before filters: 44097
2024-03-17 14:48:02,122   INFO  Total scenes after filter_info_by_object_type: 44097
2024-03-17 14:48:02,123   INFO  Total scenes after filters: 44097











































































2024-03-17 14:48:01,575   INFO  MotionTransformer(
  (context_encoder): MTREncoder(
    (agent_polyline_encoder): PointNetPolylineEncoder(
      (pre_mlps): Sequential(
        (0): Linear(in_features=30, out_features=256, bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (mlps): Sequential(
        (0): Linear(in_features=512, out_features=256, bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=256, out_features=256, bias=False)
        (4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
      )
      (out_mlps): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
      )
    )
    (map_polyline_encoder): PointNetPolylineEncoder(
      (pre_mlps): Sequential(
        (0): Linear(in_features=9, out_features=64, bias=False)
        (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=64, out_features=64, bias=False)
        (4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=64, out_features=64, bias=False)
        (7): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU()
      )
      (mlps): Sequential(
        (0): Linear(in_features=128, out_features=64, bias=False)
        (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=64, out_features=64, bias=False)
        (4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
      )
      (out_mlps): Sequential(
        (0): Linear(in_features=64, out_features=64, bias=True)
        (1): ReLU()
        (2): Linear(in_features=64, out_features=256, bias=True)
      )
    )
    (self_attn_layers): ModuleList(
      (0): TransformerEncoderLayer(
        (self_attn): MultiheadAttentionLocal(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=1024, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=1024, out_features=256, bias=True)
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
      ......
      (5): TransformerEncoderLayer(
      )
    )
  )

  # 输入数据
  *                                              num_center_objects, num_objects 会变动
  obj_trajs       torch.Size([7, 88,  11, 29])  [num_center_objects, num_objects, num_timestamps, NUM_INPUT_ATTR_AGENT]
  map_polylines   torch.Size([7, 768, 20, 9])   [num_center_objects, NUM_OF_SRC_POLYLINES, NUM_POINTS_EACH_POLYLINE, NUM_INPUT_ATTR_MAP]

  obj_trajs      = input_dict['obj_trajs'].cuda()                     # torch.Size([7, 88,  11, 29]) (num_center_objects, num_objects, num_timestamps, NUM_INPUT_ATTR_AGENT)
  obj_trajs_mask = input_dict['obj_trajs_mask'].cuda()                # torch.Size([7, 88,  11,])    (num_center_objects, num_objects, num_timestamps, )
  map_polylines      = input_dict['map_polylines'].cuda()             # torch.Size([7, 768, 20, 9])  (num_center_objects, NUM_OF_SRC_POLYLINES, NUM_POINTS_EACH_POLYLINE, NUM_INPUT_ATTR_MAP)
  map_polylines_mask = input_dict['map_polylines_mask'].cuda()        # torch.Size([7, 768, 20,])    (num_center_objects, NUM_OF_SRC_POLYLINES, NUM_POINTS_EACH_POLYLINE)

  obj_trajs_last_pos     = input_dict['obj_trajs_last_pos'].cuda()    # torch.Size([7, 88, 3,])  
  map_polylines_center   = input_dict['map_polylines_center'].cuda()  # torch.Size([7, 768, 3,])  
  track_index_to_predict = input_dict['track_index_to_predict']       # torch.Size([7])

  # Encoder
  输入如上
      obj_trajs.shape     = torch.Size([7, 88,  11, 29])
      map_polylines.shape = torch.Size([7, 768, 20, 9])

  首先各自 PointNetPolylineEncoder 输入表征
      obj_polylines_feature.shape = torch.Size([7, 88,  256])
      map_polylines_feature.shape = torch.Size([7, 768, 256])

  cat操作输入准备, 给self.apply_local_attn(), 包含两步: 
    -> KNN
    -> TransformerEncoderLayer 迭代多个
        1.输入
        global_token_feature = torch.cat((obj_polylines_feature, map_polylines_feature), dim=1) # torch.Size([7, 856, 256]) 
        global_token_mask    = torch.cat((obj_valid_mask, map_valid_mask), dim=1)               # torch.Size([7, 856])
        global_token_pos     = torch.cat((obj_trajs_last_pos, map_polylines_center), dim=1)     # torch.Size([7, 856, 3])
        2.KNN最近16条
        3.TransformerEncoderLayer
            输出维度仍然是orch.Size([7, 856, 256]) 
            再切分回原有的两部分: 
            obj_polylines_feature = global_token_feature[:, :num_objects] # torch.Size([7, 88,  256])
            map_polylines_feature = global_token_feature[:, num_objects:] # torch.Size([7, 768, 256])

  # Decoder
  输入
      center_objects_feature         # torch.Size([7, 88,  256])
      obj_feature, obj_mask, obj_pos # torch.Size([7, 88, 256]), torch.Size([7, 88]), torch.Size([7, 88, 3]))
      map_feature, map_mask, map_pos # torch.Size([7, 768, 256]), torch.Size([7, 768]), torch.Size([7, 768, 3])

  首先input projection
      对center_objects_feature, [in_proj_center_obj]
      对obj_feature,            [in_proj_obj] + [obj_decoder_layers](5 TransformerDecoderLayer)
      对map_feature,            [in_proj_map] + [map_decoder_layers](5 TransformerDecoderLayer)




  # Loss

  (motion_decoder): MTRDecoder(
    (in_proj_center_obj): Sequential(
      (0): Linear(in_features=256, out_features=512, bias=True)
      (1): ReLU()
      (2): Linear(in_features=512, out_features=512, bias=True)
    )

    (in_proj_obj): Sequential(
      (0): Linear(in_features=256, out_features=512, bias=True)
      (1): ReLU()
      (2): Linear(in_features=512, out_features=512, bias=True)
    )
    (obj_decoder_layers): ModuleList(
      (0): TransformerDecoderLayer(
        (sa_qcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_qpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_kcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_kpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (sa_v_proj): Linear(in_features=512, out_features=512, bias=True)
        (self_attn): MultiheadAttention(
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (ca_qcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_qpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_kcontent_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_kpos_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_v_proj): Linear(in_features=512, out_features=512, bias=True)
        (ca_qpos_sine_proj): Linear(in_features=512, out_features=512, bias=True)
        (cross_attn): MultiheadAttention(
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=512, bias=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
      ......
      (5): TransformerDecoderLayer(
      )
    )
  
    (in_proj_map): Sequential(
      (0): Linear(in_features=256, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=256, bias=True)
    )
    (map_decoder_layers): ModuleList(
      (0): TransformerDecoderLayer(
        (sa_qcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_qpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_kcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_kpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (sa_v_proj): Linear(in_features=256, out_features=256, bias=True)
        (self_attn): MultiheadAttention(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (ca_qcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_qpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_kcontent_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_kpos_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_v_proj): Linear(in_features=256, out_features=256, bias=True)
        (ca_qpos_sine_proj): Linear(in_features=256, out_features=256, bias=True)
        (cross_attn): MultiheadAttentionLocal(
          (out_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=1024, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=1024, out_features=256, bias=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.1, inplace=False)
        (dropout3): Dropout(p=0.1, inplace=False)
      )
      ......
      (5): TransformerDecoderLayer(
      )
    )
  
    (map_query_content_mlps): ModuleList(
      (0): Linear(in_features=512, out_features=256, bias=True)
      (1): Linear(in_features=512, out_features=256, bias=True)
      (2): Linear(in_features=512, out_features=256, bias=True)
      (3): Linear(in_features=512, out_features=256, bias=True)
      (4): Linear(in_features=512, out_features=256, bias=True)
      (5): Linear(in_features=512, out_features=256, bias=True)
    )
    (map_query_embed_mlps): Linear(in_features=512, out_features=256, bias=True)
    (obj_pos_encoding_layer): Sequential(
      (0): Linear(in_features=2, out_features=512, bias=True)
      (1): ReLU()
      (2): Linear(in_features=512, out_features=512, bias=True)
      (3): ReLU()
      (4): Linear(in_features=512, out_features=512, bias=True)
    )
    (dense_future_head): Sequential(
      (0): Linear(in_features=1024, out_features=512, bias=False)
      (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Linear(in_features=512, out_features=512, bias=False)
      (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU()
      (6): Linear(in_features=512, out_features=560, bias=True)
    )
    (future_traj_mlps): Sequential(
      (0): Linear(in_features=320, out_features=512, bias=True)
      (1): ReLU()
      (2): Linear(in_features=512, out_features=512, bias=True)
      (3): ReLU()
      (4): Linear(in_features=512, out_features=512, bias=True)
    )
    (traj_fusion_mlps): Sequential(
      (0): Linear(in_features=1024, out_features=512, bias=True)
      (1): ReLU()
      (2): Linear(in_features=512, out_features=512, bias=True)
      (3): ReLU()
      (4): Linear(in_features=512, out_features=512, bias=True)
    )
    (intention_query_mlps): Sequential(
      (0): Linear(in_features=512, out_features=512, bias=False)
      (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Linear(in_features=512, out_features=512, bias=True)
    )

    (query_feature_fusion_layers): ModuleList(
      (0): Sequential(
        (0): Linear(in_features=1280, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=True)
      )
      ......
      (5): Sequential(
      )
    )

    (motion_reg_heads): ModuleList(
      (0): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=512, out_features=560, bias=True)
      )
      ......
      (5): Sequential(
      )
    )

    (motion_cls_heads): ModuleList(
      (0): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=512, bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=512, out_features=1, bias=True)
      )
      ......
      (5): Sequential(
      )
    )
  )
)

