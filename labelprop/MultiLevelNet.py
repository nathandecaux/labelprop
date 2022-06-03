from torch import nn
import torch
import torch.nn.functional as F

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