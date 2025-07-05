import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchvision import models
import numpy as np


class VIHC_model(BaseModel):
    def name(self):
        return 'VIHC_model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.fineSize, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 2, True, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        if self.isTrain:
            input_A = input['A']
            input_B = input['B']
            if len(self.gpu_ids) > 0:
                input_A = input_A.cuda(self.gpu_ids[0], non_blocking = True)
                input_B = input_B.cuda(self.gpu_ids[0], non_blocking = True)
            self.input_A = input_A
            self.input_B = input_B
            self.image_paths = input['A_paths']
        else:
            input_A = input['A']
            if len(self.gpu_ids) > 0:
                input_A = input_A.cuda(self.gpu_ids[0], non_blocking = True)
            self.input_A = input_A
            self.image_paths = input['A_paths']    

    def forward(self):
        if self.isTrain:
            self.real_A = Variable(self.input_A)
            self.fake_B= self.netG(self.real_A)
            self.real_B = Variable(self.input_B)
        else:
            self.real_A = Variable(self.input_A)
            self.fake_B= self.netG(self.real_A)


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    # for multi scale discriminator
    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_AB_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5*self.opt.lambda_adv

        self.loss_D.backward()

    def backward_G(self):

        pred_real = self.discriminate(self.real_A, self.real_B)#, use_pool=True)#)
        pred_fake = self.netD.forward(torch.cat((self.real_A, self.fake_B), dim=1)) 

        # GAN feature matching loss
        self.loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    self.loss_G_GAN_Feat += D_weights * feat_weights * \
                    self.criterionL1(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

       
        # GAN loss (Fake Passability Loss)        
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        # VGG feature matching loss
        self.loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_vgg

        # total loss
        self.loss_G = self.loss_G_GAN + self.loss_G_GAN_Feat + self.loss_G_VGG

        # backward
        self.loss_G.backward()
        
    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG(self.real_A)
            #self.real_B = Variable(self.input_B)

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_VGG', self.loss_G_VGG.item()),
                            ('G_GAN_Feat', self.loss_G_GAN_Feat.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())
                            ])

    def get_current_visuals(self):
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('fake_B',fake_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
