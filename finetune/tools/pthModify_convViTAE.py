import os
import torch
import torchvision.models as models
import torch.nn as nn


from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '8' 

###   Padding zerp 2d   ###
# input = torch.randn(4, 5, 1, 1)
# print(input)

# pad_len1 = nn.ZeroPad2d(1)
# output = pad_len1(input)
# print(output)


# state_dict = torch.load('/home/data/jiyuqing/.tph/tph_vitdet/MAEPretrain_SceneClassification/output_dir/millionAID_224/1600_0.42_0.00015_0.05_512/checkpoint-720.pth')
state_dict = torch.load('/home/data/jiyuqing/.tph/FastConvMAE/output_dir/fastconvmae_convvitae_base_patch16/checkpoint-99.pth')
# print(state_dict)

pad_len1 = nn.ZeroPad2d(1)

# print(state_dict['model']['blocks.0.PCM.0.weight'])
state_dict['model']['blocks3.0.PCM.0.weight'] = pad_len1(state_dict['model']['blocks3.0.PCM.0.weight'])
# print(state_dict['model']['blocks.0.PCM.0.weight'])
state_dict['model']['blocks3.0.PCM.3.weight'] = pad_len1(state_dict['model']['blocks3.0.PCM.3.weight'])

state_dict['model']['blocks3.1.PCM.0.weight'] = pad_len1(state_dict['model']['blocks3.1.PCM.0.weight'])
state_dict['model']['blocks3.1.PCM.3.weight'] = pad_len1(state_dict['model']['blocks3.1.PCM.3.weight'])

state_dict['model']['blocks3.2.PCM.0.weight'] = pad_len1(state_dict['model']['blocks3.2.PCM.0.weight'])
state_dict['model']['blocks3.2.PCM.3.weight'] = pad_len1(state_dict['model']['blocks3.2.PCM.3.weight'])

state_dict['model']['blocks3.3.PCM.0.weight'] = pad_len1(state_dict['model']['blocks3.3.PCM.0.weight'])
state_dict['model']['blocks3.3.PCM.3.weight'] = pad_len1(state_dict['model']['blocks3.3.PCM.3.weight'])

state_dict['model']['blocks3.4.PCM.0.weight'] = pad_len1(state_dict['model']['blocks3.4.PCM.0.weight'])
state_dict['model']['blocks3.4.PCM.3.weight'] = pad_len1(state_dict['model']['blocks3.4.PCM.3.weight'])

state_dict['model']['blocks3.5.PCM.0.weight'] = pad_len1(state_dict['model']['blocks3.5.PCM.0.weight'])
state_dict['model']['blocks3.5.PCM.3.weight'] = pad_len1(state_dict['model']['blocks3.5.PCM.3.weight'])

state_dict['model']['blocks3.6.PCM.0.weight'] = pad_len1(state_dict['model']['blocks3.6.PCM.0.weight'])
state_dict['model']['blocks3.6.PCM.3.weight'] = pad_len1(state_dict['model']['blocks3.6.PCM.3.weight'])

state_dict['model']['blocks3.7.PCM.0.weight'] = pad_len1(state_dict['model']['blocks3.7.PCM.0.weight'])
state_dict['model']['blocks3.7.PCM.3.weight'] = pad_len1(state_dict['model']['blocks3.7.PCM.3.weight'])

state_dict['model']['blocks3.8.PCM.0.weight'] = pad_len1(state_dict['model']['blocks3.8.PCM.0.weight'])
state_dict['model']['blocks3.8.PCM.3.weight'] = pad_len1(state_dict['model']['blocks3.8.PCM.3.weight'])

state_dict['model']['blocks3.9.PCM.0.weight'] = pad_len1(state_dict['model']['blocks3.9.PCM.0.weight'])
state_dict['model']['blocks3.9.PCM.3.weight'] = pad_len1(state_dict['model']['blocks3.9.PCM.3.weight'])

state_dict['model']['blocks3.10.PCM.0.weight'] = pad_len1(state_dict['model']['blocks3.10.PCM.0.weight'])
state_dict['model']['blocks3.10.PCM.3.weight'] = pad_len1(state_dict['model']['blocks3.10.PCM.3.weight'])

torch.save(state_dict, '/home/data/jiyuqing/.tph/FastConvMAE/output_dir/fastconvmae_convvitae_base_patch16/checkpoint-99_change.pth')

print("done")