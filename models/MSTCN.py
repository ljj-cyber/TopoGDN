import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        temp = self.conv(x)
        x = self.bn(x)
        
        return x
    

class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels  # 设置输出通道数与输入通道数一致
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilations = dilations
        self.residual = residual
        self.residual_kernel_size = residual_kernel_size

        # 多尺度时间卷积分支
        self.branches = nn.ModuleList()
        for dilation in dilations:
            effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
            padding = (effective_kernel_size - 1) // 2  # 确保padding是整数
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), padding=(padding, 0), dilation=dilation),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ))

        # 最大池化和1x1卷积分支用于捕获不同的特征
        self.maxpool_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),  # 保持不变
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.conv1x1_branch = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        # 残差连接
        if self.residual:
            if self.residual_kernel_size == 1:
                self.residual_connection = nn.Identity()
            else:
                self.residual_connection = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=(residual_kernel_size, 1), stride=stride, padding=0),
                    nn.BatchNorm2d(in_channels)
                )

        # 初始化权重
        self.apply(weights_init)

    def forward(self, x):
        residual = self.residual_connection(x)

        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        branch_outputs.append(self.maxpool_branch(x))
        branch_outputs.append(self.conv1x1_branch(x))

        # 合并所有分支的输出
        out = sum(branch_outputs) / len(branch_outputs)

        # 加上残差
        if self.residual:
            out += residual

        return out


class TCN1d(nn.Module):
    def __init__(self, feature_num, kernel_size=3, dilation=1):
        super(TCN1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=feature_num, 
                               out_channels=feature_num, 
                               kernel_size=3, 
                               dilation=dilation, 
                               padding=((3-1)*dilation)//2, 
                               groups=feature_num) 
        
        self.bn1 = nn.BatchNorm1d(feature_num)

        self.conv2 = nn.Conv1d(in_channels=feature_num, 
                               out_channels=feature_num, 
                               kernel_size=5, 
                               dilation=dilation, 
                               padding=((5-1)*dilation)//2, 
                               groups=feature_num) 
        self.bn2 = nn.BatchNorm1d(feature_num)
        
        self.conv3 = nn.Conv1d(in_channels=feature_num, 
                               out_channels=feature_num, 
                               kernel_size=7, 
                               dilation=dilation, 
                               padding=((7-1)*dilation)//2, 
                               groups=feature_num)
        self.bn3 = nn.BatchNorm1d(feature_num)
        self.apply(weights_init)        

        
        
    def forward(self, x):
        # 输入x的维度为 (Batch_size, feature_num, time_step) 
        # # single scale
        # y = self.conv2(x)
        # y = self.bn2(y)
        # # y = F.relu(y)

        # Multi scale
        y1 = F.relu(self.bn1(self.conv1(x)))
        y2 = F.relu(self.bn2(self.conv2(x)))
        y3 = F.relu(self.bn3(self.conv3(x)))

        y = (y1 + y2 + y3) / 3

        return x + y


if __name__ == "__main__":
    # 示例参数
    batch_size = 32
    feature_num = 10
    time_step = 100

    # 初始化模型
    model = TCN1d(feature_num=feature_num)

    # 创建一个假的输入张量以测试
    input_tensor = torch.randn(batch_size, feature_num, time_step)

    # 前向传播
    output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
