"""Cache only frozen RGB/text condition for training; native DVS/head stay live.

This is not a closed-loop inference runtime. Every changing RGB observation is
encoded exactly once. Cache provenance includes data and frozen weight digests.
"""
import torch

from .chooser import to_pixels


@torch.no_grad()
def encode_frozen_condition(model,prepare,x,device):
    if any(p.requires_grad for n,p in model.named_parameters() if not n.startswith(('event_encoder.','vla.action_head.'))):
        raise ValueError('Frozen-condition cache requires frozen upstream encoders')
    pixels=to_pixels(x['over'],x['mosaic'],prepare,device)
    condition,features=model.vla.encode_condition(x['text'],{'pixel_values':pixels},return_distillation_features=True)
    valid=features['text_valid'];visual=condition.shape[1]-valid.shape[1]
    padding=torch.cat((torch.zeros((len(x['text']),visual),dtype=torch.bool,device=condition.device),~valid),1)
    return condition.detach(),padding.detach()


def cached_native_prediction(model,condition,padding,robot,events):
    if condition.ndim!=3 or padding.shape!=condition.shape[:2] or padding.dtype!=torch.bool:
        raise ValueError('Invalid cached RGB/text memory')
    if robot.shape!=(condition.shape[0],8) or tuple(events.shape[1:])!=(2,6,60,80):
        raise ValueError('Invalid live native inputs')
    tokens=model.event_encoder(events.float()).to(condition.dtype)
    memory=torch.cat((condition,tokens),1)
    mask=torch.cat((padding,torch.zeros((len(robot),40),dtype=torch.bool,device=condition.device)),1)
    return model.vla.action_head(memory,robot,memory_key_padding_mask=mask)


class FrozenConditionCache:
    def __init__(self,model,prepare,data,device,batch=8):
        condition=[];padding=[]
        for start in range(0,len(data),batch):
            x,_,_=data.batch(range(start,min(start+batch,len(data))))
            c,p=encode_frozen_condition(model,prepare,x,device);condition.append(c.cpu());padding.append(p.cpu())
        self.condition=torch.cat(condition);self.padding=torch.cat(padding)
        if len(self.condition)!=len(data):raise ValueError('Cache/data order mismatch')

    def batch(self,indices,device):
        ix=torch.as_tensor(indices,dtype=torch.long)
        return self.condition[ix].to(device),self.padding[ix].to(device)

    def payload(self,data_fingerprint,frozen_digest):
        return dict(format='frozen_rgb_text_condition_v108',dataset_sha256=data_fingerprint,frozen_upstream_sha256=frozen_digest,condition=self.condition,padding=self.padding,scope='Frozen upstream only; DVS/native action head recomputed on every optimizer step')
