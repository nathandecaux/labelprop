from torch import nn
import torch
import torch.nn.functional as F

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
        self.final_conv=nn.Conv2d(in_channels*(levels+1),in_channels,kernel_size=3,padding=1)    
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


    def forward(self,x,registration=False):
        """
        For each levels, downsample the input and apply the convolutional block.
        """
        x_levels=[x]
        for downsampling in self.downsample_blocks:
            x_downsampled = downsampling(x_levels[-1])
            x_levels.append(x_downsampled)

        for i in range(len(x_levels)):
            x_levels[i]=self.conv_blocks[i](x_levels[i])
  
        # return torch.stack(x_levels,dim=0).mean(0)
        return self.final_conv(torch.cat(x_levels,dim=1))