from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_

# DWDconv
class DepthwiseSeparableDilatedConv(nn.Module):
    """
    DWDConv

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        dilation: 膨胀率（空洞率）
        stride: 步长
        padding: 填充
        bias: 是否使用偏置
        activation: 激活函数类型
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 dilation=1, 
                 stride=1, 
                 padding=None,
                 bias=False,
                 activation='relu'):
        super(DepthwiseSeparableDilatedConv, self).__init__()
        
        if padding is None:
            padding = dilation * (kernel_size - 1) // 2
        
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        # 深度空洞卷积
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # 逐点卷积
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        return x

class DepthWiseConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, pad=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel, stride, pad, groups=in_c),
            nn.GroupNorm(4, in_c),
            nn.Conv2d(in_c, out_c, 1)
        )
    def forward(self, x):
        return self.net(x)

# LDGA
class LightweightDGA(nn.Module):
    def __init__(self, in_c, out_c, k_size=3):
        super(LightweightDGA, self).__init__()
        
        self.dwconv = DepthWiseConv2d(
            in_c, in_c, 
            kernel=k_size, 
            stride=1, 
            pad=k_size//2
        )
        
        # 空间注意力生成
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_c, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # 最终输出变换
        self.output_conv = DepthWiseConv2d(in_c, out_c, kernel=1, stride=1, pad=0)
        
        # 残差连接
        self.residual_conv = None
        if in_c != out_c:
            self.residual_conv = DepthWiseConv2d(in_c, out_c, kernel=1, stride=1, pad=0)
    
    def forward(self, x):
        identity = x
        
        # 深度可分离卷积特征提取
        out = self.dwconv(x)
        
        # 空间注意力
        attention = self.spatial_attention(out)
        out = out * attention
        
        # 输出变换
        out = self.output_conv(out)
        
        # 残差连接
        if self.residual_conv is not None:
            identity = self.residual_conv(identity)
        
        out = out + identity
        return out

# DWD模块
class DWDBlock(nn.Module):
    def __init__(self, channels, dilation=2):
        super().__init__()
        self.conv1 = DepthwiseSeparableDilatedConv(channels, channels, dilation=dilation)
        self.conv2 = DepthwiseSeparableDilatedConv(channels, channels, dilation=1)
        
    def forward(self, x):
        identity = x
        out = F.gelu(self.conv1(x))
        out = self.conv2(out)
        return F.gelu(out + identity)

# LCAB
class Lightweight_Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list):
        super().__init__()
        c_list_sum = sum(c_list)
        self.c_list = c_list
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 全局特征提取
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        # self.get_all_att = nn.Linear(c_list_sum, c_list_sum)
        
        # 单一共享全连接层
        self.shared_fc = nn.Linear(c_list_sum, c_list_sum)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, t1, t2, t3, t4, t5):
        # 全局平均池化
        att = torch.cat((self.avgpool(t1), 
                         self.avgpool(t2), 
                         self.avgpool(t3), 
                         self.avgpool(t4), 
                         self.avgpool(t5)), dim=1)
        
        # 全局特征交互
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        att = att.transpose(-1, -2)
        
        # 共享全连接层处理
        att = self.shared_fc(att.squeeze(-1))
        att = self.sigmoid(att)
        
        # 动态分割输出
        start_idx = 0
        att_list = []
        for c in self.c_list:
            att_split = att[:, start_idx:start_idx+c]
            att_split = att_split.unsqueeze(-1).unsqueeze(-1)
            att_list.append(att_split)
            start_idx += c
        
        att1 = att_list[0].expand_as(t1)
        att2 = att_list[1].expand_as(t2)
        att3 = att_list[2].expand_as(t3)
        att4 = att_list[3].expand_as(t4)
        att5 = att_list[4].expand_as(t5)
        
        return att1, att2, att3, att4, att5

# LSAB
class Lightweight_Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            
            att = F.pad(att, (3, 3, 3, 3), mode='reflect')
            att = self.shared_conv2d(att)
            att_list.append(att)
            
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]

class LSCAB(nn.Module):
    """
    ightweight SCAB
    
    1. 参数共享式通道交互机制
    2. 标准化空间注意力重构  
    3. 并行注意力融合
    """
    def __init__(self, c_list):
        super().__init__()
        
        # 轻量化通道和空间注意力模块
        self.catt = Lightweight_Channel_Att_Bridge(c_list)
        self.satt = Lightweight_Spatial_Att_Bridge()
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, t1, t2, t3, t4, t5):
        identity = [t1, t2, t3, t4, t5]
        
        # 并行计算通道和空间注意力
        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        
        # 并行注意力融合
        channel_atts = [catt1, catt2, catt3, catt4, catt5]
        spatial_atts = [satt1, satt2, satt3, satt4, satt5]
        
        outputs = []
        for i, (t, c_att, s_att) in enumerate(zip(identity, channel_atts, spatial_atts)):
            # 并行加权融合
            combined_att = self.alpha * c_att + self.beta * s_att
            output = combined_att * t + t  # 残差连接
            outputs.append(output)
        
        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]


class MAUNet(nn.Module):
    """
    MAUNet
    使用DWD、LDGA、LSCAB等轻量化模块
    """
    def __init__(self, num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True):
        super().__init__()
        
        self.bridge = bridge

        # 编码器DWDConv
        self.encoder1 = DepthwiseSeparableDilatedConv(input_channels, c_list[0], dilation=1)
        
        # 深层编码器 - DWD + LDGA
        self.encoder2 = nn.Sequential(DWDBlock(c_list[0]), LightweightDGA(c_list[0], c_list[1]))
        self.encoder3 = nn.Sequential(DWDBlock(c_list[1]), LightweightDGA(c_list[1], c_list[2]))
        self.encoder4 = nn.Sequential(DWDBlock(c_list[2]), LightweightDGA(c_list[2], c_list[3]))
        self.encoder5 = nn.Sequential(DWDBlock(c_list[3]), LightweightDGA(c_list[3], c_list[4]))
        self.encoder6 = nn.Sequential(DWDBlock(c_list[4]), LightweightDGA(c_list[4], c_list[5]))

        # LSCAB桥接模块
        if bridge:
            self.lscab = LSCAB(c_list[:5])
            print('LSCAB was used')
        
        # 解码器 - LDGA + DWD
        self.decoder1 = nn.Sequential(LightweightDGA(c_list[5], c_list[4]), DWDBlock(c_list[4]))
        self.decoder2 = nn.Sequential(LightweightDGA(c_list[4], c_list[3]), DWDBlock(c_list[3]))
        self.decoder3 = nn.Sequential(LightweightDGA(c_list[3], c_list[2]), DWDBlock(c_list[2]))
        self.decoder4 = nn.Sequential(LightweightDGA(c_list[2], c_list[1]), DWDBlock(c_list[1]))
        self.decoder5 = nn.Sequential(LightweightDGA(c_list[1], c_list[0]), DWDBlock(c_list[0]))

        # 批归一化
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        # 最终输出层
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x, return_features):
        # 存储中间特征
        features = []

        # 编码阶段
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4
        
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        if return_features:
            features.extend([t1, t2, t3, t4, t5])

        # LSCAB桥接
        if self.bridge:
            t1, t2, t3, t4, t5 = self.lscab(t1, t2, t3, t4, t5)
        
        # 瓶颈层
        out = F.gelu(self.encoder6(out))

        # 解码阶段
        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        out5 = torch.add(out5, t5)  # 跳跃连接
        
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2,2), mode='bilinear', align_corners=True))
        out4 = torch.add(out4, t4)
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2,2), mode='bilinear', align_corners=True))
        out3 = torch.add(out3, t3)
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2,2), mode='bilinear', align_corners=True))
        out2 = torch.add(out2, t2)
        
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2,2), mode='bilinear', align_corners=True))
        out1 = torch.add(out1, t1)

        # 最终输出
        out0 = F.interpolate(self.final(out1), scale_factor=(2,2), mode='bilinear', align_corners=True)
        
        if return_features:
            return torch.sigmoid(out0), features
        else:
            return torch.sigmoid(out0)

# 示例
if __name__ == "__main__":
    # 创建MAUNet模型
    model = MAUNet()
    
    # 测试前向传播
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 参数量统计
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"MAUNet参数量: {count_parameters(model):,}")
    
    # 模型结构概览
    print("\nMAUNet模型结构:")
    print("编码器: 深度可分离空洞卷积 + DWD + LDGA")
    print("桥接: LSCAB轻量化跨阶段注意力")
    print("解码器: LDGA + DWD + 深度可分离空洞卷积")