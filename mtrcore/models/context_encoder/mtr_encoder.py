# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import torch
import torch.nn as nn


from mtrcore.models.utils.transformer import transformer_encoder_layer, position_encoding_utils
from mtrcore.models.utils import polyline_encoder
from mtrcore.utils import common_utils
from mtrcore.ops.knn import knn_utils


class MTREncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        # build polyline encoders
        self.agent_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_AGENT + 1,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_AGENT,
            out_channels=self.model_cfg.D_MODEL
        )
        self.map_polyline_encoder   = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_MAP,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_MAP,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_MAP,
            num_pre_layers=self.model_cfg.NUM_LAYER_IN_PRE_MLP_MAP,
            out_channels=self.model_cfg.D_MODEL
        )

        # build transformer encoder layers
        pe_self_attn_layers   = []
        self.use_local_attn   = self.model_cfg.get('USE_LOCAL_ATTN', False)
        self.num_out_channels = self.model_cfg.D_MODEL # for decoder

        for _ in range(self.model_cfg.NUM_ATTN_LAYERS):
            pe_self_attn_layers.append(
                self.build_transformer_encoder_layer(
                    d_model = self.model_cfg.D_MODEL,
                    nhead   = self.model_cfg.NUM_ATTN_HEAD,
                    dropout = self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
                    normalize_before = False,
                    use_local_attn   = self.use_local_attn
                )
            )
        self.self_attn_layers = nn.ModuleList(pe_self_attn_layers)


    # """对Agent和Map 分别进行PointNetPolylineEncoder"""
    def build_polyline_encoder(
        self,
        in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None
    ):
        ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels
        )
        return ret_polyline_encoder

    # """For self.self_attn_layers, in 进而用于apply_global_attn/local..."""
    def build_transformer_encoder_layer(
        self,
        d_model, nhead, dropout=0.1, normalize_before=False, use_local_attn=False
    ):
        single_encoder_layer = transformer_encoder_layer.TransformerEncoderLayer(
            d_model = d_model,
            nhead   = nhead,
            dim_feedforward = d_model * 4,
            dropout = dropout,
            normalize_before = normalize_before,
            use_local_attn   = use_local_attn
        )
        return single_encoder_layer

    
    def apply_global_attn(self, x, x_mask, x_pos):
        """ Args:
            x      (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos  (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)
        batch_size, N, d_model = x.shape
        x_t      = x.permute(1, 0, 2)
        x_mask_t = x_mask.permute(1, 0, 2)
        x_pos_t  = x_pos.permute(1, 0, 2)
 
        # positional encoding
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(
            x_pos_t, hidden_dim = d_model
        )

        # global attn
        for k in range(len(self.self_attn_layers)):
            x_t = self.self_attn_layers[k](
                src = x_t,
                src_key_padding_mask =~ x_mask_t,
                pos = pos_embedding
            )
        x_out = x_t.permute(1, 0, 2)  # (batch_size, N, d_model)
        return x_out


    # 同上述_global_.基础self.self_attn_layers之上; 使用knn_utils.knn_batch_mlogk; 
    def apply_local_attn(self, x, x_mask, x_pos, num_of_neighbors):
        """ Args:
            x      (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos  (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)
        batch_size, N, d_model = x.shape

        x_stack_full     = x.view(-1, d_model)  # (batch_size * N, d_model)
        x_pos_stack_full = x_pos.view(-1, 3)
        batch_idxs_full  = torch.arange(batch_size).type_as(x)[:, None].repeat(1, N).view(-1).int()  # (batch_size * N)
        x_mask_stack     = x_mask.view(-1)

        # filter invalid elements
        x_stack     = x_stack_full[x_mask_stack]
        x_pos_stack = x_pos_stack_full[x_mask_stack]
        batch_idxs  = batch_idxs_full[x_mask_stack]
        # KNN
        batch_offsets = common_utils.get_batch_offsets(batch_idxs=batch_idxs, bs=batch_size).int()  # (batch_size + 1)
        batch_cnt = batch_offsets[1:] - batch_offsets[:-1]

        index_pair = knn_utils.knn_batch_mlogk(
            x_pos_stack, x_pos_stack,  batch_idxs, batch_offsets, num_of_neighbors
        )  # (num_valid_elems, K)

        # positional encoding
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(
            x_pos_stack[None, :, 0:2], hidden_dim=d_model
        )[0]

        # local attn
        output = x_stack
        for k in range(len(self.self_attn_layers)):
            output = self.self_attn_layers[k](
                src = output,
                pos = pos_embedding,
                index_pair = index_pair,
                query_batch_cnt = batch_cnt,
                key_batch_cnt = batch_cnt,
                index_pair_batch = batch_idxs
            )

        ret_full_feature = torch.zeros_like(x_stack_full)  # (batch_size * N, d_model)
        ret_full_feature[x_mask_stack] = output

        ret_full_feature = ret_full_feature.view(batch_size, N, d_model)
        return ret_full_feature



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
              input_dict:
        """
        input_dict = batch_dict['input_dict']
        
        obj_trajs      = input_dict['obj_trajs'].cuda()                     # torch.Size([7, 17,  11, 29]) (num_center_objects, num_objects, num_timestamps, NUM_INPUT_ATTR_AGENT)
        obj_trajs_mask = input_dict['obj_trajs_mask'].cuda()                # torch.Size([7, 17,  11,])    (num_center_objects, num_objects, num_timestamps, )
        map_polylines      = input_dict['map_polylines'].cuda()             # torch.Size([7, 768, 20, 9])  (num_center_objects, NUM_OF_SRC_POLYLINES, NUM_POINTS_EACH_POLYLINE, NUM_INPUT_ATTR_MAP)
        map_polylines_mask = input_dict['map_polylines_mask'].cuda()        # torch.Size([7, 768, 20,])    (num_center_objects, NUM_OF_SRC_POLYLINES, NUM_POINTS_EACH_POLYLINE)

        obj_trajs_last_pos     = input_dict['obj_trajs_last_pos'].cuda()    # torch.Size([7, 17, 3,])  
        map_polylines_center   = input_dict['map_polylines_center'].cuda()  # torch.Size([7, 768, 3,])  
        track_index_to_predict = input_dict['track_index_to_predict']       # torch.Size([7])

        assert obj_trajs_mask.dtype == torch.bool and map_polylines_mask.dtype == torch.bool

        num_center_objects, num_objects,   num_timestamps, _ = obj_trajs.shape
        _,                  num_polylines,  _,             _ = map_polylines.shape
        # num_polylines = map_polylines.shape[1]

        # apply polyline encoder
        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1) # torch.Size([7, 17, 11, 30]) (num_center_objects, num_objects, num_timestamps, NUM_INPUT_ATTR_AGENT+1)
        obj_polylines_feature = self.agent_polyline_encoder(obj_trajs_in, obj_trajs_mask)               # torch.Size([7, 17, 256])    (num_center_objects, num_objects, C)
        map_polylines_feature = self.map_polyline_encoder(map_polylines, map_polylines_mask)            # torch.Size([7, 768, 256])   (num_center_objects, num_polylines, C)

        # apply self-attn
        obj_valid_mask = (obj_trajs_mask.sum(dim=-1) > 0)                                               # torch.Size([7, 17]  (num_center_objects, num_objects)
        map_valid_mask = (map_polylines_mask.sum(dim=-1) > 0)                                           # torch.Size([7, 768] (num_center_objects, NUM_OF_SRC_POLYLINES)

        # cat()
        global_token_feature = torch.cat((obj_polylines_feature, map_polylines_feature), dim=1)         # torch.Size([7, 785, 256]) 
        global_token_mask    = torch.cat((obj_valid_mask, map_valid_mask), dim=1)                       # torch.Size([7, 785])
        global_token_pos     = torch.cat((obj_trajs_last_pos, map_polylines_center), dim=1)             # torch.Size([7, 785, 3])

        if self.use_local_attn:
            global_token_feature = self.apply_local_attn(
                x      = global_token_feature,
                x_mask = global_token_mask,
                x_pos  = global_token_pos,
                num_of_neighbors = self.model_cfg.NUM_OF_ATTN_NEIGHBORS
            )
        else:
            global_token_feature = self.apply_global_attn(
                x      = global_token_feature,
                x_mask = global_token_mask,
                x_pos  = global_token_pos,
            )                                                                                           # global_token_feature torch.Size([7, 785, 256])

        # split()
        obj_polylines_feature = global_token_feature[:, :num_objects]                                   # torch.Size([7, 17, 256])
        map_polylines_feature = global_token_feature[:, num_objects:]                                   # torch.Size([7, 768, 256])
        assert map_polylines_feature.shape[1] == num_polylines

        # organize return features
        center_objects_feature = obj_polylines_feature[torch.arange(num_center_objects), track_index_to_predict]

        batch_dict['center_objects_feature'] = center_objects_feature
        batch_dict['obj_feature'] = obj_polylines_feature                                               # torch.Size([7, 256])
        batch_dict['map_feature'] = map_polylines_feature                                               # torch.Size([7, 17, 256])
        batch_dict['obj_mask'] = obj_valid_mask                                                         # torch.Size([7, 17])
        batch_dict['map_mask'] = map_valid_mask                                                         # torch.Size([7, 768])
        batch_dict['obj_pos'] = obj_trajs_last_pos                                                      # torch.Size([7, 17, 3]))
        batch_dict['map_pos'] = map_polylines_center                                                    # torch.Size([7, 768, 3])

        import pdb; pdb.set_trace()
        return batch_dict

""" 
num_center_objects, num_objects [7, 17] 会变化

obj_trajs_last_pos = obj_trajs[:, :, -1, 0:3] 对应是obj_trajs历史轨迹最后时间帧对应的pos

map_polylines中 NUM_OF_SRC_POLYLINES=768 是当前Batch下距离num_center_objects最近的768条polyline ---> def create_map_data_for_center_objects() in waymo_dataset.py


input_dict['obj_trajs'].shape               torch.Size([7, 17, 11, 29])     [num_center_objects, num_objects, num_timestamps, NUM_INPUT_ATTR_AGENT ]
                                                                            17个object, 7个center_object
                                                                            
input_dict['obj_trajs_mask'].shape          torch.Size([7, 17, 11])         [num_center_objects, num_objects, num_timestamps ]
                                                                            是否valid
                                                                            
input_dict['map_polylines'].shape           torch.Size([7, 768, 20, 9])     [num_center_objects, NUM_OF_SRC_POLYLINES, NUM_POINTS_EACH_POLYLINE, NUM_INPUT_ATTR_MAP ]

input_dict['map_polylines_mask'].shape      torch.Size([7, 768, 20])        [num_center_objects, NUM_OF_SRC_POLYLINES, NUM_POINTS_EACH_POLYLINE ]

input_dict['obj_trajs_last_pos'].shape      torch.Size([7, 17, 3])          [num_center_objects, num_objects, 3]                                   
input_dict['map_polylines_center'].shape    torch.Size([7, 768, 3])         [num_center_objects, NUM_OF_SRC_POLYLINES, 3]
input_dict['track_index_to_predict'].shape  torch.Size([7])                 [num_center_objects]

input_dict['scenario_id'].shape
input_dict['obj_trajs_pos'].shape
input_dict['obj_types'].shape
input_dict['obj_ids'].shape
input_dict['center_objects_world'].shape
...

see. MTR/mtr/datasets/dataset.py
        Args:
        batch_list:
            scenario_id: (num_center_objects)
            track_index_to_predict (num_center_objects):

            obj_trajs                   (num_center_objects, num_objects, num_timestamps, num_attrs):
            obj_trajs_mask              (num_center_objects, num_objects, num_timestamps):
            map_polylines               (num_center_objects, num_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask          (num_center_objects, num_polylines, num_points_each_polyline)

            obj_trajs_pos:              (num_center_objects, num_objects, num_timestamps, 3)
            obj_trajs_last_pos:         (num_center_objects, num_objects, 3)
            obj_types:                  (num_objects)
            obj_ids:                    (num_objects)

            center_objects_world:       (num_center_objects, 10)  [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            center_objects_type:        (num_center_objects)
            center_objects_id:          (num_center_objects)

            obj_trajs_future_state      (num_center_objects, num_objects, num_future_timestamps, 4): [x, y, vx, vy]
            obj_trajs_future_mask       (num_center_objects, num_objects, num_future_timestamps):
            center_gt_trajs             (num_center_objects, num_future_timestamps, 4): [x, y, vx, vy]
            center_gt_trajs_mask        (num_center_objects, num_future_timestamps):
            center_gt_final_valid_idx   (num_center_objects): the final valid timestamp in num_future_timestamps
"""