# coding=utf-8
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


import torch
import torch.nn as nn



#######################################################################
#----------------Generator-Resnet-with Skip Connection----------------#
#######################################################################
class VihcGAN(nn.Module):
    def __init__(self, input_dim, img_size=224, output_dim=3, vis=False):
        super(VihcGAN, self).__init__()
        output_nc = output_dim
        ngf = 64
        use_bias = False
        activation = nn.ReLU(True)
        norm_layer = nn.InstanceNorm2d
        padding_type = 'reflect'
        mult = 4

        #--------------------------------------Encoder---------------------------------------------#
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_dim, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        setattr(self, 'encoder_1', nn.Sequential(*model))
        #--------------------------------------------------------#
        n_downsampling = 2
        model = []
        i = 0
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_2', nn.Sequential(*model))
        #--------------------------------------------------------#
        model = []
        i = 1
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_3', nn.Sequential(*model))
        #--------------------------------------------------------#
        model = []
        i = 2
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_4', nn.Sequential(*model))
        #--------------------------------------------------------#
        model = []
        i = 3
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_5', nn.Sequential(*model))
       
        #-----------------------------------Resnet Blocks------------------------------------------#
                
        self.res_1 = ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_2 = ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_3 = ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_4 = ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_5 = ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_6 = ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_7 = ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_8 = ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_9 = ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
 
        #--------------------------------------Decoder---------------------------------------------#
        model = []
        model = [nn.ConvTranspose2d(ngf * mult * 2, ngf * mult, #1024，512
                             kernel_size=3, stride=2,
                             padding=1, output_padding=1,
                             bias=use_bias),
          norm_layer(ngf * mult),
          nn.ReLU(True)]

        setattr(self, 'decoder_1', nn.Sequential(*model))
        #--------------------------------------------------------#
        model = []
        model = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult/2), #1024,256
                             kernel_size=3, stride=2,
                             padding=1, output_padding=1,
                             bias=use_bias),
          norm_layer(int(ngf * mult/2)),
          nn.ReLU(True)]

        setattr(self, 'decoder_2', nn.Sequential(*model))
        #--------------------------------------------------------#
        mult = int(mult/2)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult/2), #512,128
                             kernel_size=3, stride=2,
                             padding=1, output_padding=1,
                             bias=use_bias),
          norm_layer(int(ngf * mult/2)),
          nn.ReLU(True)]

        setattr(self, 'decoder_3', nn.Sequential(*model))
        #--------------------------------------------------------#     
        mult = int(mult/2)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult/2), #256,64
                             kernel_size=3, stride=2,
                             padding=1, output_padding=1,
                             bias=use_bias),
          norm_layer(int(ngf * mult/2)),
          nn.ReLU(True)]

        setattr(self, 'decoder_4', nn.Sequential(*model))
        #--------------------------------------------------------#
        model = []
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, 'decoder_5', nn.Sequential(*model))

        
    def forward(self, x):

        #--------encoder--------#
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)        
        x4 = self.encoder_4(x3)        
        x5 = self.encoder_5(x4)            
   
        #-------bottleneck-------#
        x = self.res_1(x5)
        x = self.res_2(x)     
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)
        x = self.res_6(x)
        x = self.res_7(x)
        x = self.res_8(x)
        x = self.res_9(x) 

        #--------decoder--------#
        x6 = self.decoder_1(x)  
        x6 = torch.cat((x6,x4), dim=1)

        x7 = self.decoder_2(x6) 
        x7 = torch.cat((x7,x3), dim=1)

        x8 = self.decoder_3(x7) 
        x8 = torch.cat((x8,x2), dim=1)

        x9 = self.decoder_4(x8) 

        x = self.decoder_5(x9)
        return x

#############################################################
#-----------------------Resnet Block------------------------#
#############################################################
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

        setattr(self, 'resnet block', nn.Sequential(*conv_block))
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out