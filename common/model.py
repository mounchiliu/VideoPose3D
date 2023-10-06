# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn

class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()
        
        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'
        
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths
        
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        self.pad = [ filter_widths[0] // 2 ]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out*3, 1)
        

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum
            
    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2*frames
    
    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames
        
    def forward(self, x):
        # 模型forward函数入口， reshape x
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
        
        sz = x.shape[:3]  # batch, 243, 17, 2
        x = x.view(x.shape[0], x.shape[1], -1)  # batch, 243, 17*2
        x = x.permute(0, 2, 1)  # batch, 17*2, 243
        
        x = self._forward_blocks(x)  # 调用对应子类的函数（TemporalModelOptimized1f or TemporalModel）
        
        x = x.permute(0, 2, 1)  # reshape x 以便于输出
        x = x.view(sz[0], -1, self.num_joints_out, 3) # 变成坐标的形式e.g. 17 x 3
        
        return x    

class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        
        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        
        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            
            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x):
        # 将输入的多帧2D关节点，每帧的关节点投影到高维  1 x 34 x 2941 -> 1 x 1024 x 2941
        # 1 for 1 data in batch, 即对应1个视频，2941对应经过pad后该视频一共有2941帧
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x)))) # k=3,s=1, s=1因为这里是同时处理多帧pose
        
        for i in range(len(self.pad) - 1):  # for each temporal block
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            #
            # res: 对应下面进行卷积的结果，即取第一个中心帧和最后一个中心帧间的所有帧（因为这两个帧之间的其他所有帧都可以做3组数据的中心帧）
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            # self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            # 这个模型用于test， 与train model的区别在于，layers_conv为空洞卷积，但其内部参数与train使用的正常卷积一致
            # test中的空洞卷积，使得前后间隔d帧的信息合在一起提取特征，最后计算当前帧的3D pose
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.shrink(x)  # shrink: 使用conv k=1,变换维度至17x3
        return x
    
class TemporalModelOptimized1f(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.
    
    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)

        # expand: conv, change feature channel and size
        # 这里对时间序列组成的点进行1维卷积
        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], stride=filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []

        # use causal convolutions instead of symmetric convolutions
        # for real-time applications use causal convolutions
        self.causal_shift = [ (filter_widths[0] // 2) if causal else 0 ] # 2
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)  # frame padding
            self.causal_shift.append((filter_widths[i]//2) if causal else 0)

            # layers_con依次记录每一个block里包含的所有conv、
            # layers_bn依次记录每一个block里包含的bn
            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False)) # 3
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x):
        # 将输入的多帧2D关节点，每帧的关节点投影到高维  1024(batch) x 34 x 243(num_frame) -> 1024 x 1024 x 81
        # expand: conv, change feature channel and size
        # 这里对时间序列组成的点进行1维卷积: k = 3, stride = 3, 因此每3个时间序列为一组进行卷积操作，提取特征值
        # 同理下面的卷积，k=3s = 3, 即每3个一组（每组之间帧没有重复）计算特征值
        # 下面还有一个卷积是k=1,s=1,channel input = channel out， 用于使用卷积的conv对输入x进行调整
        # 而test时，stride=1,通过调节d来实现对间隔的数据进行卷积， 注意d是每个数据间的间隔，stride s为每组输入kernel数据间的间隔
        # 可以通过paper中图1看出，所有蓝色点对应train时，s=3生成的结果
        # 这个操作可以在test时，使用dilated conv with s = 1 对应
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))  # s=3, 处理单帧数据，to avoid generating unused intermediate results
        
        for i in range(len(self.pad) - 1):
            # single-frame batching
            # This implementation replaces dilated convolutions with stride convolutions
            # to avoid generating unused intermediate results.
            # res为按照x的最后一个维度进行数据提取（即时间维度）， 若self.filter_widths = k，则每间隔3个提取一次
            # 这里res取的时当前层每组卷积的中间帧。 k=3,每3个时间的数据为一组做卷积，得到的结果可以理解为这3个的中心帧的特帧
            # 所以使用res把处理前和处理后的连接起来
            res = x[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]

            # skip connection: 一部分channel的x直接输出，一部分进入卷积层
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))  # 对所有x进行卷积、bn、relu、drop
            # 按照论文Fig 2，一共两次卷积操作（一次k=3,一次k=1），所以下一行又写了一次
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))  # res connection
        
        x = self.shrink(x)  # shrink: 使用conv, k=1变换维度: 输出所有batch的坐标，一个batch对应一个输出， batch x 51
        return x