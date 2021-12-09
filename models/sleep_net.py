import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
from ops.base_module import *
from ops.base_ops import ConsensusModule
from ops.transforms import *


class SleepNet(nn.Module):
    def __init__(self, num_segments=5, base_mode=None, pretrained=True, **kwargs):
        super(SleepNet, self).__init__()

        if '50' in base_model:
            resnet_model = fbresnet50(num_segments, pretrained)
        else:
            resnet_model = fbresnet101(num_segments, pretrained)

        self.model = TDN(resnet_model, alpha=0.5, beta=0.5)
                    


    def forward(self, x):
        return x 
    

class TDN(nn.Module):
    def __init__(self, resnet_model, alpha, beta):
        super(TDN, self).__init__()

        # conv1
        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)

        # conv1 motion + pooling
        self.conv1_motion = nn.Sequential(nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        params = [x.clone() for x in list(resnet_model.children())[0].parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + [12,] + kernel_size[2:]
        new_kernel = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        self.conv1_motion[0].weight.data = new_kernel
        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.resnext_layer1 = nn.Sequential(*list(resnet_model.children())[4])
        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])
        self.layer2_bak = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3_bak = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4_bak = nn.Sequential(*list(resnet_model.children())[7])
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc = list(resnet_model.children())[8]
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        """
        Takes one segment and returns 2D motion feature map
        """
        # sampled frames
        x1, x2, x3, x4, x5 = x[:,0:3,:,:], x[:,3:6,:,:], x[:,6:9,:,:], x[:,9:12,:,:], x[:,12:15,:,:]
        
        # compute temporal RGB differences in low resolution and upsample again as in Short-term TDM
        x_motion = self.conv1_5(self.avg_diff(torch.cat([x2-x1, x3-x2, x4-x3, x5-x4,1).view(-1, 12, x.size()[2], x.size()[3])
        x_motion = self.maxpool_diff(x_motion)

        # conv1 + pooling + motion 
        x = self.conv1(x3)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x_motion_1 = F.interpolate(x_motion, x.size()[2:])
        x = self.alpha * x + self.beta * x_motion_1

        # res2
        x = self.layer1_bak(x)
        x_motion_2 = self.resnext_layer1(x_motion)
        x_motion_2 = F.interpolate(x_motion_2, x.size()[2:])
        x = self.alpha * x + self.beta * x_motion_2
        
        x = self.layer2_bak(x)
        x = self.layer3_bak(x)
        x = self.layer4_bak(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x       


class TSN(nn.Module):
    def __init__(self,
                 num_class,
                 num_segments, frames_per_segment=None,
                 base_model='resnet50', dropout=0.8, img_feature_dim=256, crop_num=1,
                 partial_bn=True, print_spec=True, pretrain='imagenet'):
        super(TSN, self).__init__()
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type 
        self.img_feature_dim = img_feature_dim
        self.pretrain = pretrain
        self.base_model_name = base_model
        self.target_transforms = {86:87, 87:86, 93:94, 94:93, 166:167, 167:166} # ?

        if print_spec:
            print(("""
                Initializing TSN with base model: {}.
                TSN Configurations:
                    input_modality:     {}
                    num_segments:       {}
                    new_length:         {}
                    consensus_module:   {}
                    dropout_ratio:      {}
                    img_feature_dim:    {}
                """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))

        self.prepare_base_model(base_model, num_segments)
        feature_dim = self.prepare_tsn(num_class)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def prepare_base_model(self, base_model, num_segments):
        print('=> base model: {}'.format(base_model))
        if 'resnet' in base_model:
            self.base_model = tdn_net(base_model, num_segments)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def prepare_tsn(self, num_class):
        """
        add dropout before the last fc layer of the base model if self.dropout!=0, and initialize
        """
        std = 0.001
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout==0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)
        return feature_dim

    def forward(self, input):
        channels = 3 * self.frames_per_segment # RGB * frames per segment
        base_out = self.base_model(input.view((-1, channels * self.num_segments) + input.size()[-2:])) # (B, C, H, W) 

        if self.dropout>0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)

            return output.squeeze(1)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(TSN, self).train(mode)
        count = 0
        if self.enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        else:
            print("No BN layer Freezing.")

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_weight = []
        bn = []
        custom_ops = []
        inorm = []
        conv_cnt = 0
        bn_cnt = 0

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt==1:
                    first_conv_weight.append(ps[0])
                    if len(ps)==2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps)==2:
                        normal_bias.append(ps[1])
            elif isinstance(m. torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps)==2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        if self.fc_lr5: # fine_tuning for UCF/HMDB
            return [
                {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
                'name': "first_conv_weight"},
                {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
                'name': "first_conv_bias"},
                {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
                'name': "normal_weight"},
                {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
                'name': "normal_bias"},
                {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
                'name': "BN scale/shift"},
                {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
                'name': "custom_ops"},
                {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
                'name': "lr5_weight"},
                {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
                'name': "lr10_bias"},
            ]
        else : # default 
            return [
                {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
                'name': "first_conv_weight"},
                {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
                'name': "first_conv_bias"},
                {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
                'name': "normal_weight"},
                {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
                'name': "normal_bias"},
                {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
                'name': "BN scale/shift"},
                {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
                'name': "custom_ops"},
            ]

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
        else:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                            GroupRandomHorizontalFlip_sth(self.target_transforms)])
















































                





















    











































 
        

     



