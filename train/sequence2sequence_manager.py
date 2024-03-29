import numpy 
import torch
from typing import List, Dict
class FieldsSequence:
    """
    sequence is a list of metadata
    # [
    #   timestamp_1:{'Field':Field, 'stamp_status':stamp_status]},
    #   timestamp_2:{'Field':Field, 'stamp_status':stamp_status]}
    #     ...............................................
    #   timestamp_n:{'Field':Field, 'stamp_status':stamp_status]}
    # ]
    """
    input_channel_num = None
    target_channel_num = None

    def __init__(self, args):
        self.B           = args.Train.batch_size,
        self.image_shape = args.Model.model.img_size
        self.input_len   = args.Model.model.history_length
        self.pred_len    = args.Model.model.pred_len
        
        self.inputs_sequence = [None]*self.input_len
        self.target_sequence = []
    
    def initial_unnormilized_inputs_field(self, inputs_sequence):
        assert len(inputs_sequence) == self.input_len
        _=[self._stamp_check(t) for t in inputs_sequence]
        self.inputs_sequence = inputs_sequence
        
    
    def _stamp_check(self, stamp):
        assert isinstance(stamp,dict), "each stamp should be dict {'field':Field, 'stamp_status':stamp_status]}"
        #assert stamp['field'].shape[2:] == (*self.image_shape, ), f"stamp['field'].shape[2:]={stamp['field'].shape[2:]} != self.image_shape={self.image_shape}"


    def normalize_a_field(self, unnormlized_field):
        # input should b (B,P,L,W,H) or (B,P,W,H) or {'field': (B,P,W,H) }
        return unnormlized_field

    def unnormalized_a_field(self, normlized_field):
        return normlized_field

    def push_unnormilized_target_field(self, target_sequence):
        if target_sequence is None:
            self.target_sequence = None
            self.key_for_one_stamp = {'field'}
        assert len(target_sequence) == self.pred_len
        _=[self._stamp_check(t) for t in target_sequence]
        self.target_sequence += target_sequence
        self.key_for_one_stamp = set(target_sequence[0].keys())

    def push_a_normlized_field(self, preded_normlized_fields:torch.Tensor):
        """
        Input is only a dict {'field': Preded_Field (B, P, L=1, W, H)} 
        """
        preded_normlized_fields = self.unnormalized_a_field(preded_normlized_fields)
        if len(preded_normlized_fields.shape) == len(self.image_shape) + 1 + 1 + 1:   # (B, P, L, H, W)
            pred_lens = preded_normlized_fields.shape[2]
            preded_normlized_fields = preded_normlized_fields.split(pred_lens,dim=2) # (B, P, H, W)
        elif len(preded_normlized_fields.shape) == len(self.image_shape) + 1 + 1 :  
            preded_normlized_fields = [preded_normlized_fields]
            pred_lens = 1
        else:
            raise ValueError(f"preded_normlized_fields.shape[2:]={preded_normlized_fields.shape[2:]} != (self.pred_len,*self.image_shape)={self.pred_len,*self.image_shape}")
        #_=[self._stamp_check(t) for t in preded_normlized_fields]
        
        new_candidate = [{}]*pred_lens
        for key in self.key_for_one_stamp: # <-- only update the field
            if key == 'field':
                for i in range(pred_lens):new_candidate[i]['field'] = preded_normlized_fields[i]
            else:
                for i in range(pred_lens):new_candidate[i][key] = self.target_sequence[i][key]
            
        self.inputs_sequence = self.inputs_sequence[-self.pred_len:] + new_candidate
        self.target_sequence = self.target_sequence[self.pred_len:]


    def concat_dict(self, sequence: List[Dict]):
        inputs_dict = {}
        for stamp in sequence:
            for key, val in stamp.items():
                if key not in inputs_dict:inputs_dict[key] = []
                inputs_dict[key].append(self.normalize_a_field(val))
        for key in inputs_dict.keys():
            inputs_dict[key] = torch.stack(inputs_dict[key], 2) # --> (B,P,L,W,H)
        return inputs_dict
    
    @property
    def inputs(self):
        ## concat along the time dimension
        if len(self.inputs_sequence) == 1:return self.inputs_sequence[0]
        return self.concat_dict(self.inputs_sequence)

    @property
    def target(self):
        if len(self.target_sequence) == 1:return self.target_sequence[0]
        return self.concat_dict(self.target_sequence)

    def get_inputs_and_target(self):
        return self.inputs, self.target
    


class FieldsSequenceWithChannelShifting(FieldsSequence):
    def __init__(self, config):
        super().__init__(config)
        self.train_channel_from_this_stamp = config.get('train_channel_from_this_stamp', None)
        self.train_channel_from_next_stamp = config.get('train_channel_from_next_stamp', None)
        self.pred_channel_for_next_stamp   = config.get('pred_channel_for_next_stamp', None)
    
    def get_inputs_and_target(self):
        _inputs = self.inputs
        if self.train_channel_from_this_stamp:
            assert len(self.inputs['field'].shape) == 4
            self.inputs['field'] = self.inputs['field'][:, self.train_channel_from_this_stamp]

        _target = self.target
        if self.train_channel_from_next_stamp:
            assert len(self.inputs['field'].shape) == 4
            self.inputs['field'] = torch.cat([self.inputs['field'], 
                                              self.target['field'][:, self.train_channel_from_next_stamp]], 1)
        return _inputs, _target
    
    def push_a_normlized_field(self, preded_normlized_fields):
        
        preded_normlized_fields = self.unnormalized_a_field(preded_normlized_fields['field'])
        pred_lens = preded_normlized_fields.shape[2]
        preded_normlized_fields = preded_normlized_fields.split(pred_lens,dim=2)
        _=[self._stamp_check(t) for t in preded_normlized_fields]
        
        new_candidate = [{}]*pred_lens
        for key in self.key_for_one_stamp:
            if key == 'field':
                for i in range(pred_lens):
                    if self.pred_channel_for_next_stamp:
                        next_tensor = self.target_sequence[i]['field'].clone() #<-- do not change the metadata
                        next_tensor[:, self.pred_channel_for_next_stamp] = preded_normlized_fields[i]
                    else:
                        next_tensor = preded_normlized_fields[i]
                    new_candidate[i]['field'] = next_tensor
            else:
                for i in range(pred_lens):new_candidate[i][key] = self.target_sequence[i][key]
            
        self.inputs_sequence = self.inputs_sequence[-self.pred_len:] + new_candidate
        self.target_sequence = self.target_sequence[self.pred_len:]
