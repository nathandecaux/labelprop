import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from monai.networks.nets import UNet,Classifier,DenseNet,SwinUNETR,DynUNet,AttentionUnet
import numpy as np
import math
from kornia.geometry.transform import get_perspective_transform,get_affine_matrix2d
from .MultiLevelNet import MultiLevelNet as MLNet
class VxmDense(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
                 inshape,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 sub_levels=3):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        self.inshape=inshape
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        self.levels=sub_levels+1
        # configure core unet model
        # self.unet_model = Unet(
        #     inshape,
        #     infeats=(src_feats + trg_feats)
        # )
        filters=  [16, 32, 32, 32]
        #DynUNet(spatial_dims, in_channels, out_channels, kernel_size, strides, upsample_kernel_size, filters=None, dropout=None, norm_name=('INSTANCE', {'affine': True}), act_name=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}), deep_supervision=False, deep_supr_num=1, res_block=False, trans_bias=False)
        # self.unet_model=UNet(src_feats*2,trg_feats*2,16,filters,strides=(len(filters)-1)*[2],num_res_units=(len(filters)-2))
        self.unet_model=UNet(src_feats*2,trg_feats*2,16,filters,strides=(len(filters)-1)*[2],num_res_units=(len(filters)-2))

        # self.unet_model=DynUNet(2,src_feats*2,trg_feats*2,(len(filters))*[(3,3)],strides=(len(filters))*[2],upsample_kernel_size=(len(filters)-1)*[(3,3)])
        # self.unet_model=AttentionUnet(2,src_feats*2,16,channels=filters,strides=(len(filters)-1)*[2],dropout=0.1)
        # self.unet_model=SwinUNETR(
        #     in_channels=src_feats*2,
        #     out_channels=trg_feats*2,
        #     img_size=inshape,
        #     feature_size=24,
        #     spatial_dims=2,
        # )
        # self.unet_model=BasicUNet(2,src_feats*2,2,features=filters,upsample='nontrainable')
        # self.unet_model=MultiLevelNet(inshape=inshape,levels=sub_levels)
        # self.unet_model=MLNet(inshape=inshape,levels=sub_levels)

        # self.unet_model=SingleLevelNet(inshape)
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)
        # ☺init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        # probabilities are not supported in pytorch

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape,levels=sub_levels+1)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        flow_field = self.unet_model(x)
        # transform into flow field
        flow_field = self.flow(flow_field)
        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
        # warp image with flow field

        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, pos_flow, preint_flow)
        else:
            if self.bidir:
                return y_source, pos_flow, neg_flow
            else:
                return y_source, pos_flow


class FeaturesToAffine(nn.Module):
    """
    Dense network that takes pixels of features map and convert it to affine matrix 
    """

    def __init__(self,inshape):
        super().__init__()
        
        self.inshape=inshape
        self.outshape=4*4  
        self.conv1=nn.Conv2d(16,1,kernel_size=1)
        self.flatten=nn.Flatten()
        self.layer=nn.Sequential(nn.Linear(self.inshape,self.outshape),nn.ReLU())
        self.layer2=nn.Sequential(nn.Linear(self.outshape,self.outshape),nn.Tanh())
    def forward(self,x):
        x=self.conv1(x)
        x=self.layer(self.flatten(x))
        x=self.layer2(x)
        return x

class AffineGenerator(nn.Module):
    """
    Dense network that takes affine matrix and generate affine transformation
    """

    def __init__(self,inshape):
        super().__init__()
        self.inshape=inshape
        self.network=Classifier(inshape,9,(2,2,2,2,2),(2,2,2,2,2))
        self.max_angle=nn.Parameter(30.*torch.ones(1))
        self.max_scale=nn.Parameter(0.01*torch.ones(1))
    def forward(self,x1,x2):
        x=torch.cat([x1,x2],dim=1)
        x=self.network(x)
        # x=get_affine_matrix2d(translations=nn.Tanh()(x[:,0:2])*self.inshape[0]/2,center=nn.Tanh()(x[:,5:7])*self.inshape[0]/2,scale=nn.Tanh()(x[:,2:4])*self.max_scale+1,angle=nn.Tanh()(x[:,4])*self.max_angle)
        x=get_affine_matrix2d(nn.Tanh()(x[:,0]),center=x[:,5:7],scale=x[:,2:4],angle=x[:,4])

        return x

class AffineGenerator3D(nn.Module):
    """
    Dense network that takes affine matrix and generate affine transformation
    """

    def __init__(self,inshape):
        super().__init__()
        self.inshape=(2,inshape[0],inshape[1],inshape[2])
        print(self.inshape)
        #self.network=Classifier(inshape,16,(2,2,2,2,2),(2,2,2,2,2))
        self.network=DenseNet(3,2,16)
    def forward(self,x1,x2):
        x=torch.cat([x1,x2],dim=1)
        x=self.network(x)
        
        return x.view(-1,4,4)
    

class SingleLevelNet(nn.Module):
    """
    Convolutional network generating deformation field
    """

    def __init__(self,inshape, in_channels=2,features=16):
        super().__init__()
        print('Initializing MultiLevelNet')
        self.inshape=inshape
        self.in_channels=in_channels
        self.conv_blocks=self.get_conv_blocks(in_channels,features)
    
    def get_conv_blocks(self, in_channels, intermediate_features):
        """
        For each levels, create a convolutional block with two Conv Tanh BatchNorm layers
        """
        # return nn.Sequential(nn.BatchNorm2d(in_channels),
        #                                 nn.LeakyReLU(0.2),
        #                                 nn.Conv2d(in_channels, intermediate_features, kernel_size=3, padding=1),
        #                                 nn.BatchNorm2d(intermediate_features),
        #                                 nn.LeakyReLU(0.2),
        #                                 nn.Conv2d(intermediate_features, intermediate_features, kernel_size=3, padding=1),
        #                                 nn.BatchNorm2d(intermediate_features),
        #                                 nn.LeakyReLU(0.2),
        #                                 nn.Conv2d(intermediate_features, in_channels, kernel_size=3, padding=1),
        #                                 nn.LeakyReLU(0.2))
        return nn.Sequential(
            nn.BatchNorm2d(intermediate_features),
            nn.Conv2d(in_channels, intermediate_features, kernel_size=3, padding=1)
        )
    def forward(self,x):
        """
        Forward pass of the network
        Args:
            x ([Tensor]): Tensor of shape (B,C,H,W)
        Returns:
            [Tensor]: Tensor of shape (B,C,H,W)
        """
        x=self.conv_blocks(x)
        print(x.shape)
        return x.view(-1,self.in_channels,x.shape[-2],x.shape[-1])

class MultiLevelNet(nn.Module):
    """
    Convolutional network generating deformation field with different scales.
    """

    def __init__(self,inshape, in_channels=2, levels=3,features=16):
        super().__init__()
        print('Initializing MultiLevelNet')
        self.inshape=inshape
        self.levels=levels
        self.in_channels=in_channels
        self.downsample_blocks=self.get_downsample_blocks(in_channels,levels)
        self.shapes=[int(self.inshape[0]/(i+1)) for i in range(levels+1)]
        self.conv_blocks=self.get_conv_blocks(in_channels,levels,features)
        # self.transformers=self.get_transformer_list(levels,inshape)
    
    def get_downsample_blocks(self, in_channels, levels):
        blocks = nn.ModuleList()
        for i in range(levels):
            blocks.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2))
        return blocks
    
    def get_conv_blocks(self, in_channels, levels,intermediate_features):
        """
        For each levels, create a convolutional block with two Conv Tanh BatchNorm layers
        """
        blocks = nn.ModuleList()
        for i in range(levels+1):
            blocks.append(nn.Sequential(
                            nn.BatchNorm2d(in_channels),
                            nn.Conv2d(in_channels, intermediate_features, kernel_size=3, padding=1),
                            nn.Tanh(),
                            nn.BatchNorm2d(intermediate_features),
                            nn.Conv2d(intermediate_features, in_channels, kernel_size=3, padding=1),
                            nn.Upsample(self.inshape, mode='bilinear',align_corners=True)))
        return blocks

    def get_transformer_list(self,levels,inshape):
        """
        Create a list of spatial transformer for each level.
        """
        transformers = nn.ModuleList()
        for i in range(levels):
            transformers.append(SpatialTransformer((inshape[0]/(2**i),inshape[1]/(2**i))))
        return transformers

    def compose_list(self,flows):
        flows=list(flows)
        compo=flows[-1]
        for flow in reversed(flows[:-1]):
            compo=self.compose_deformation(flow,compo)
        return compo
    def compose_deformation(self,flow_i_k,flow_k_j):
        """ Returns flow_k_j(flow_i_k(.)) flow
        Args:
            flow_i_k 
            flow_k_j
        Returns:
            [Tensor]: Flow field flow_i_j = flow_k_j(flow_i_k(.))
        """        
        flow_i_j= flow_k_j+self.transformer(flow_i_k,flow_k_j)
        return flow_i_j

    def forward(self,x,registration=False):
        """
        For each levels, downsample the input and apply the convolutional block.
        """
        x_levels=[x]
        for downsampling in self.downsample_blocks:
            x_downsampled = downsampling(x_levels[-1])
            x_levels.append(x_downsampled)
        # for i in range(len(self.conv_blocks)):
        #     x_conv = self.conv_blocks[i](x_levels[-1])
        #     x_levels.append(x_conv)
        # #For each x_levels,interpolate to the original resolution, and sum x_levels
        # for i in range(1,len(x_levels)):
        #     x_levels[i]=F.interpolate(x_levels[i],size=self.inshape,mode='bilinear')
        for i in range(len(x_levels)):
            x_levels[i]=self.conv_blocks[i](x_levels[i])
  
        return torch.stack(x_levels,dim=0).mean(0)



    

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear',levels=4):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        # grid= torch.cat([grid]*levels,dim=0)
        self.levels=levels
        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        # if src.shape[0]==1:
        #     src=src.repeat(self.levels,1,1,1)
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

import torch
import torch.nn.functional as F
import numpy as np
import math


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred, mean=True):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(y_pred.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        if mean:
            return -torch.mean(cc)
        else:
            return cc


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad