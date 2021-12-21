### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F
import copy
import torch.nn.utils.spectral_norm as spectral_norm
import re

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        # print('sss', m.weight)
        #m.weight.data.normal_(1.0, 0.02)
        #m.bias.data.fill_(0)
        pass

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':    
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)       
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'unet':
        netG = UnetGenerator(input_nc, output_nc, 3, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[])
    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[], netD='multi'):
    if netD == 'multi':        
        norm_layer = get_norm_layer(norm_type=norm)   
        netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    elif netD == 'hand':
        netD = NLayerDiscriminator(input_nc, ndf=64, n_layers=n_layers_D, norm_layer=nn.BatchNorm2d, use_sigmoid=False, 
                            getIntermFeat=False, addname='hand')
        #netD = HandDiscriminator(42)
    else:
        raise('discriminator not implemented!')
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        '''model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model
        # print "hello friends", len(list(model_global))    
        # print [model_global[i] for i in range(len(list(model_global)))]
        # print "minus 3 here", len(list(model_global))-3    
        # print [model_global[i] for i in range(len(list(model_global))-3)]
        model_global = [model_global[i] for i in range(len(list(model_global))-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)       '''

        # spade config
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer)
        self.compute_latent_vector_size = model_global.compute_latent_vector_size
        self.global_sw, self.global_sh = self.compute_latent_vector_size()
        self.global_fc = model_global.fc
        self.global_head_0 = model_global.head_0
        self.global_G_middle_0 = model_global.G_middle_0
        self.global_G_middle_1 = model_global.G_middle_1
        self.global_up_0 = model_global.up_0
        self.global_up_1 = model_global.up_1
        self.global_up_2 = model_global.up_2
        self.global_up_3 = model_global.up_3
        self.global_conv_img = model_global.conv_img
        self.global_up = model_global.up
        print('n_local_enhancers: ', n_local_enhancers)

        ''' ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample_spade_resnet = []
            for i in range(n_blocks_local):
                # model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]
                model_upsample_spade_resnet += [SPADEResnetBlock(ngf_global * 2, ngf_global * 2, input_nc)]

            ### upsample
            model_upsample = []
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample_spade_resnet))
            setattr(self, 'model' + str(n) + '_3', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False) '''

        # Spade variation
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            print(ngf, ngf_global)
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), nn.ReLU(True),
                                nn.Conv2d(ngf_global, 3, kernel_size=3, stride=2, padding=1), nn.ReLU(True)]
            setattr(self, 'model_downsample_' + str(n), nn.Sequential(*model_downsample))

            ### residual blocks
            sw, sh = self.compute_latent_vector_size()
            final_nc = ngf_global
            setattr(self, 'spade_' + str(n) + '_sw', sw - (2 ** (n_local_enhancers - n)))
            setattr(self, 'spade_' + str(n) + '_sh', sh - (2 ** (n_local_enhancers - n)))
            setattr(self, 'spade_' + str(n) + '_fc', nn.Conv2d(input_nc, 16 * ngf_global, 3, padding=1))
            setattr(self, 'spade_' + str(n) + '_head_0', SPADEResnetBlock(16 * ngf_global, 16 * ngf_global, input_nc))
            setattr(self, 'spade_' + str(n) + '_G_middle_0', SPADEResnetBlock(16 * ngf_global, 16 * ngf_global, input_nc))
            setattr(self, 'spade_' + str(n) + '_G_middle_1', SPADEResnetBlock(16 * ngf_global, 16 * ngf_global, input_nc))
            setattr(self, 'spade_' + str(n) + '_up_0', SPADEResnetBlock(16 * ngf_global, 8 * ngf_global, input_nc))
            setattr(self, 'spade_' + str(n) + '_up_1', SPADEResnetBlock(8 * ngf_global, 8 * ngf_global, input_nc))
            setattr(self, 'spade_' + str(n) + '_up_2', SPADEResnetBlock(4 * ngf_global, 8 * ngf_global, input_nc))
            setattr(self, 'spade_' + str(n) + '_up_3', SPADEResnetBlock(2 * ngf_global, 8 * ngf_global, input_nc))
            setattr(self, 'spade_' + str(n) + '_conv_img', nn.Conv2d(final_nc, 3, (3, 3), padding=1))
            setattr(self, 'spade_' + str(n) + '_up', nn.Upsample(scale_factor=2))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        ''' ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample_spade_resnet = getattr(self, 'model'+str(n_local_enhancers)+'_2')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_3')
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]
            temp = model_downsample(input_i) + output_prev
            output_spade = model_upsample_spade_resnet(temp, input)
            output_prev = model_upsample(output_spade)
        return output_prev '''

        # spade variation
        seg = input

        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        x = F.interpolate(seg, size=(self.global_sh, self.global_sw))
        x = self.global_fc(x)
        x = self.global_head_0(x, seg)
        x = self.global_up(x)

        x = self.global_G_middle_0(x, seg)
        x = self.global_G_middle_1(x, seg)

        x = self.global_up(x)
        x = self.global_up_0(x, seg)
        x = self.global_up(x)
        x = self.global_up_1(x, seg)
        x = self.global_up(x)
        x = self.global_up_2(x, seg)
        x = self.global_up(x)
        x = self.global_up_3(x, seg)

        x = self.global_conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        print(x.shape, seg.shape)

        for n in range(1, self.n_local_enhancers + 1):
            input_i = input_downsampled[self.n_local_enhancers - n]
            model_downsample = getattr(self, 'model_downsample_' + str(n))
            output = model_downsample(input_i)
            print(output.shape, x.shape)
            x += output
            sh = getattr(self, 'spade_' + str(n) + '_sh')
            sw = getattr(self, 'spade_' + str(n) + '_sw')
            fc = getattr(self, 'spade_' + str(n) + '_fc')
            head_0 = getattr(self, 'spade_' + str(n) + '_head_0')
            up = getattr(self, 'spade_' + str(n) + '_up')
            G_middle_0 = getattr(self, 'spade_' + str(n) + '_G_middle_0')
            G_middle_1 = getattr(self, 'spade_' + str(n) + '_G_middle_1')
            up_0 = getattr(self, 'spade_' + str(n) + '_up_0')
            up_1 = getattr(self, 'spade_' + str(n) + '_up_1')
            up_2 = getattr(self, 'spade_' + str(n) + '_up_2')
            up_3 = getattr(self, 'spade_' + str(n) + '_up_3')
            conv_img = getattr(self, 'spade_' + str(n) + '_conv_img')

            x = F.interpolate(seg, size=(sh, sw))
            x = fc(x)
            x = head_0(x, seg)
            x = up(x)

            x = G_middle_0(x, seg)
            x = G_middle_1(x, seg)

            x = up(x)
            x = up_0(x, seg)
            x = up(x)
            x = up_1(x, seg)
            x = up(x)
            print(x.shape, seg.shape)
            x = up_2(x, seg)
            x = up(x)
            x = up_3(x, seg)

            x = conv_img(F.leaky_relu(x, 2e-1))
            x = F.tanh(x)

        return x


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)
        self.n_blocks = n_blocks

        #model1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        '''for i in range(n_downsampling):
            mult = 2**i
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]'''

        ### resnet blocks
        mult = 2**n_downsampling
        model2 = []
        '''for i in range(n_blocks):
            # model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            #model2 += [SPADEResnetBlock(ngf * mult, ngf * mult, input_nc)]
            # model2 = SPADEResnetBlock(ngf * mult, ngf * mult, input_nc)
            setattr(self, 'model_spade_' + str(i), SPADEResnetBlock(ngf * mult, ngf * mult, input_nc))'''

        
        ### upsample
        '''model3 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model3 += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model1 = nn.Sequential(*model1)
        # self.model2 = nn.Sequential(*model2)
        # self.model2 = model2
        self.model3 = nn.Sequential(*model3)'''

        ngf = 20
        # print(ngf)
        self.sw, self.sh = self.compute_latent_vector_size()

        self.fc = nn.Conv2d(input_nc, 16 * ngf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * ngf, 16 * ngf, input_nc)

        self.G_middle_0 = SPADEResnetBlock(16 * ngf, 16 * ngf, input_nc)
        self.G_middle_1 = SPADEResnetBlock(16 * ngf, 16 * ngf, input_nc)

        self.up_0 = SPADEResnetBlock(16 * ngf, 8 * ngf, input_nc)
        self.up_1 = SPADEResnetBlock(8 * ngf, 4 * ngf, input_nc)
        self.up_2 = SPADEResnetBlock(4 * ngf, 2 * ngf, input_nc)
        self.up_3 = SPADEResnetBlock(2 * ngf, 1 * ngf, input_nc)

        final_nc = ngf
        self.conv_img = nn.Conv2d(final_nc, 3, (3, 3), padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self):
        num_up_layers = 5
        crop_size = 512
        aspect_ratio = 2
        sw = crop_size // (2 ** num_up_layers)
        sh = round(sw / aspect_ratio)
        return sw, sh
            
    def forward(self, input):
        seg = input
        '''x1 = self.model1(input)
        x2 = x1
        for i in range(self.n_blocks):
            x2 = getattr(self, 'model_spade_' + str(i))(x2, input)
        return self.model3(x2)'''

        x = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.fc(x)
        x = self.head_0(x, seg)
        x = self.up(x)

        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x
        
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super(SPADE, self).__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        #elif param_free_norm_type == 'syncbatch':
            #self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=(ks,ks), padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=(ks,ks), padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=(ks,ks), padding=pw)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, input_nc):
        super(SPADEResnetBlock, self).__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=(3,3), padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=(3,3), padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=(1,1), bias=False)

        # apply spectral norm if specified
        '''if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)'''

        # define normalization layers
        #spade_config_str = opt.norm_G.replace('spectral', '')
        spade_config_str = 'spadeinstance3x3'
        self.norm_0 = SPADE(spade_config_str, fin, input_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, input_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, input_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()        
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            indices = (inst == i).nonzero() # n x 4            
            for j in range(self.output_nc):
                output_ins = outputs[indices[:,0], indices[:,1] + j, indices[:,2], indices[:,3]]                    
                mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                outputs_mean[indices[:,0], indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                        
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False, addname=''):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.addname = addname

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, addname + 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            # self.model = nn.Sequential(*sequence_stream)
            setattr(self, addname + 'model', nn.Sequential(*sequence_stream))

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, self.addname + 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            # return self.model(input) 
            model = getattr(self, self.addname + 'model')
            return model(input)

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)       

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    
    
class HandDiscriminator(nn.Module):
    
    def __init__(self, pose_dim):
        
        super(HandDiscriminator, self).__init__()
        
        self.disc = torch.nn.Sequential(
            torch.nn.Linear(in_features=pose_dim, out_features=240, bias=True),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=240, out_features=240, bias=True),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=240, out_features=1, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)
