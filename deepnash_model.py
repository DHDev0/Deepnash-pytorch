import torch
import torch.nn as nn


class conv_resblock(nn.Module):
    def __init__(self,
                 chanels_in,
                 channels_out,
                 strides):
        super(conv_resblock, self).__init__()
        
        self.s = strides
        d2_convolution_0 = nn.Conv2d(in_channels = chanels_in, out_channels = channels_out//2, kernel_size = 3, stride=strides, padding=1)
        relu= nn.ReLU()
        d2_convolution_1 = nn.Conv2d(in_channels = channels_out//2, out_channels = channels_out, kernel_size = 3, stride=1, padding=1)
        
        if self.s > 1 :# when applying striding, residual connections are also processed by a convolution layer with 1 ×1 kernel
            self.d2_convolution_2 = nn.Conv2d(in_channels = chanels_in, out_channels = channels_out, kernel_size =1, stride=strides, padding=0)
            
        
        conv_r = [d2_convolution_0,
                  relu,
                  d2_convolution_1,
                  relu]
        self.conv_r = nn.Sequential(*tuple(conv_r))
        
    def forward(self, x_0):
        x = self.conv_r(x_0)
        if self.s > 1 :
            x_0 = self.d2_convolution_2(x_0)
        output = x + x_0 # when applying striding, residual connections are also processed by a convolution layer with 1 ×1 kernel
        return output, x_0 # output and skipt-out


class deconv_resblock(nn.Module):
    def __init__(self,
                 chanels_in,
                 channels_out,
                 strides,
                 parity_test = True):
        super(deconv_resblock, self).__init__()
        
        self.s = strides
        relu= nn.ReLU()
        
        if parity_test: 
            d2_deconvolution_0 = nn.ConvTranspose2d(in_channels = chanels_in, out_channels = channels_out//2, kernel_size = 3, stride=strides, padding=1,output_padding = 1 if strides > 1 else 0)
            d2_deconvolution_1 = nn.ConvTranspose2d(in_channels = chanels_in, out_channels = channels_out//2, kernel_size = 1, stride=strides, padding=0 ,output_padding = 1 if strides > 1 else 0)
            d2_deconvolution_2 = nn.ConvTranspose2d(in_channels = channels_out//2, out_channels = channels_out, kernel_size = 3, stride=1, padding=1)
            self.r2 = nn.ConvTranspose2d(in_channels = chanels_in, out_channels = channels_out, kernel_size = 3, stride=strides, padding=1,output_padding = 1 if strides > 1 else 0)
        else:
            d2_deconvolution_0 = nn.ConvTranspose2d(in_channels = chanels_in, out_channels = channels_out//2, kernel_size = 3, stride=strides, padding=1,output_padding = 0 )
            d2_deconvolution_1 = nn.ConvTranspose2d(in_channels = chanels_in, out_channels = channels_out//2, kernel_size = 1, stride=strides, padding=0 ,output_padding = 0 )
            d2_deconvolution_2 = nn.ConvTranspose2d(in_channels = channels_out//2, out_channels = channels_out, kernel_size = 3, stride=1, padding=1)
            self.r2 = nn.ConvTranspose2d(in_channels = chanels_in, out_channels = channels_out, kernel_size = 3, stride=strides, padding=1,output_padding = 0 )
        

        
        
        self.deconv_0 = nn.Sequential(d2_deconvolution_0,relu)
        self.deconv_1 = d2_deconvolution_1
        self.deconv_2 = nn.Sequential(d2_deconvolution_2, relu)
        
    def forward(self, x_0 ,skip_in):
        x = self.deconv_0( x_0 )
        residual_conv = self.deconv_1(skip_in)
        x = x + residual_conv # nn + residual skip_in
        x_0 = self.r2(x_0)
        output = self.deconv_2(x) + x_0 # nn + residual
        return output 

class Pyramid_module(nn.Module):
    def __init__(self,
                 outer_block : int = 0, 
                 inner_block : int = 0,
                 channels = 256,
                 parity_test = True):
        super(Pyramid_module, self).__init__()
        
        d2_convolution = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, stride=1, padding=1)
        relu = torch.nn.ReLU() 
        
        self.layer_0 = nn.Sequential(d2_convolution, relu)
        self.layer_1 = ([conv_resblock(channels,channels,1)] * outer_block )
        self.layer_2 = conv_resblock(channels,int(channels*1.25),2)
        self.layer_3 = ([conv_resblock(int(channels*1.25),int(channels*1.25),1)] * inner_block )
        self.layer_4 = ([deconv_resblock(int(channels*1.25),int(channels*1.25),1,parity_test = parity_test)] * inner_block )
        self.layer_5 = deconv_resblock(int(channels*1.25),channels,2,parity_test = parity_test)
        self.layer_6 = ([deconv_resblock(channels,channels,1,parity_test = parity_test)] * outer_block )   
        
    def forward(self, x):
        # 2D Convolution + Relu
        x = self.layer_0(x)

        #Conv Resblock (outer blocks)
        layer_1 = {}
        for i in range(len(self.layer_1)):
            x , layer_1[f"skip_out_{i}"] = self.layer_1[i](x)
        
        #Conv Resblock
        x , layer_2_residual = self.layer_2(x)
        
        #Conv Resblock (inner blocks)
        layer_3 = {}
        for i in range(len(self.layer_3)):
            x , layer_3[f"skip_out_{i}"] = self.layer_3[i](x)
        
        #Deconv Resblock (inner blocks)
        for i in range(len(self.layer_4)):
            x = self.layer_4[i](x , layer_3[f"skip_out_{i}"])
        
        #Deconv Resblock
        x = self.layer_5(x , layer_2_residual)
        
        #Deconv Resblock (outer blocks)
        for i in range(len(self.layer_6)):
            x  = self.layer_6[i](x,layer_1[f"skip_out_{i}"])
        
        return x
        
class Masked_Softmax(nn.Module):
    def __init__(self):
        super(Masked_Softmax, self).__init__()
        self.softmax = nn.Softmax(1)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.float()
            x_masked = x * mask + (1 - 1 / mask)
        else:
            x_masked = x
            
        x_max = x_masked.max(-1,keepdim=True).values[0]
        x_exp = (x - x_max).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(-1,keepdim=True)

class Deepnash_model(nn.Module):
    def __init__(self,
                 channels = 256 , 
                 policy_width = 10 ,
                 policy_height = 10):
        super(Deepnash_model, self).__init__()
        
        assert all(isinstance(val, int) for val in [channels,policy_width,policy_height])
        if policy_width % 2 == 0 and policy_height % 2 == 0: parity_test = 1
        elif policy_width % 2 != 0 and policy_height % 2 != 0: parity_test = 0
        else: raise Exception("policy_width and policy_height need to bee both odd or pair")
        assert int(channels * 1.25) == channels * 1.25, f" channels = {channels} must be an integer when multiplied by 1.25 "
        
        parity_test = policy_width % 2 == 0 and policy_height % 2 == 0 
        self.init_pyramid = Pyramid_module(outer_block = 2, inner_block = 2)
        self.policy_head =  nn.Sequential(
                                          Pyramid_module(outer_block = 1, inner_block = 0, channels = channels , parity_test = parity_test),
                                          nn.Conv2d(in_channels = channels, out_channels = 1, kernel_size = 3, stride=1, padding=1),
                                          nn.ReLU(),
                                          Masked_Softmax()
                                         )
        self.value_head =  nn.Sequential(
                                         Pyramid_module(outer_block = 0, inner_block = 0, channels = channels , parity_test = parity_test),
                                         nn.Conv2d(in_channels = channels, out_channels = 1, kernel_size = 3, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.Flatten(1),
                                         nn.Linear(in_features = policy_width*policy_height , out_features = 1)
                                         )
    
    def policy(self, obs, mode=None):
        if mode == "Observation": # "Phase_specific_observation" without the first pyramid module
            obs = self.init_pyramid(obs)
        policy_h = self.policy_head(obs)        
        return policy_h
    
    def value(self, obs, mode=None):
        if mode == "Observation": # "Phase_specific_observation" without the first pyramid module
            obs = self.init_pyramid(obs)
        value_h = self.value_head(obs)
        return value_h

      
# #test
# tensor_test = torch.rand(1, 256, 10 , 10)
# deepnash = Deepnash_model(channels = 256 , 
#                           policy_width = 10 ,
#                           policy_height = 10)
# policy = deepnash.policy(tensor_test,mode = "Observation")
# value = deepnash.value(tensor_test,mode = "Observation")
# print(policy,value)
