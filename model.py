import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import math

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def Union(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    return 1

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # self.mode=mode

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if mode=='val':
        #     print("cancel BN...................")
        #     norm_layer = nn.Identity
        # self.mode=mode
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        before_pool_feature=x
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x,before_pool_feature

    def forward(self, x):
        return self._forward_impl(x)


class gat(nn.Module):
    def __init__(self, device,pretrained,out_channels=48, num_layers=3, num_heads=8,
                 dropout=0.5,feature_dim=2048,use_layer_norm=True, use_residual=True, use_residual_linear=False, mode='train'):
        super(gat, self).__init__()
        torch.cuda.manual_seed(233)
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3],num_classes=feature_dim//2)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=True)
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            self.backbone.load_state_dict(state_dict, strict=False)
        self.feature_dim=feature_dim
        self.mode=mode
        self.device=device
        self.in_channels=feature_dim
        self.hidden_channels=feature_dim//8
        self.out_channels=out_channels
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.dropout=dropout
        self.layers = torch.nn.ModuleList()
        kwargs = {'bias':False,'share_weights':True}
        self.layers.append(GATv2Conv(self.in_channels, self.hidden_channels // num_heads, num_heads, **kwargs))
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.use_residual_linear = use_residual_linear
        self.layer_norms = torch.nn.ModuleList()
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(self.hidden_channels))
        self.residuals = torch.nn.ModuleList()
        if use_residual_linear and use_residual:
            self.residuals.append(nn.Linear(self.in_channels, self.hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(
                GATv2Conv(self.hidden_channels, self.hidden_channels // num_heads, num_heads, **kwargs))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(self.hidden_channels))
            if use_residual_linear and use_residual:
                self.residuals.append(nn.Linear(self.hidden_channels, self.hidden_channels))
        self.layers.append(GATv2Conv(self.hidden_channels, out_channels//num_heads, num_heads, **kwargs ))#out_channels=48
        if use_residual_linear and use_residual:
            self.residuals.append(nn.Linear(self.hidden_channels, out_channels))
        self.dropout = dropout
        self.non_linearity = F.relu
        self.fc2=nn.Linear(44*out_channels,feature_dim)
        if mode=='train':
            self.conv1=nn.Conv2d(feature_dim, feature_dim//2, kernel_size=7, stride=1, padding=3)
            self.bn1 = nn.BatchNorm2d(feature_dim//2)
            self.relu = nn.ReLU(inplace=True)
            self.conv2=nn.Conv2d(feature_dim//2, feature_dim//4, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(feature_dim//4)
            self.conv3=nn.Conv2d(feature_dim//4, feature_dim//8, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(feature_dim//8)
            self.conv4=nn.Conv2d(feature_dim//8, feature_dim//16, kernel_size=3, stride=1, padding=1)
            self.bn4 = nn.BatchNorm2d(feature_dim//16)
            self.conv5=nn.Conv2d(feature_dim//16, feature_dim//32, kernel_size=3, stride=1, padding=1)
            self.bn5 = nn.BatchNorm2d(feature_dim//32)
            self.conv6=nn.Conv2d(feature_dim//32, 3, kernel_size=7, stride=1, padding=3)
        print(f"learnable_params: {sum(p.numel() for p in list(self.parameters()) if p.requires_grad)}")

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.layer_norms:
            layer.reset_parameters()
        for layer in self.residuals:
            layer.reset_parameters()

    def forward(self, block):
        seq=block[0]
        masks_list=block[1]
        points_list=block[2]
        masks_class_list=block[3]
        V=block[4]
        edge_info=block[5]
        (B, n, H, W) = masks_list.shape
        (B,C,H,W)=seq.shape
        R_feature,_=self.backbone(seq)
        R_feature=R_feature.unsqueeze(1).expand(B,n,self.feature_dim//2)
        mblock=seq.unsqueeze(1).expand(B, n, C, H, W)
        masks_list=masks_list.unsqueeze(2).expand(B, n, C, H, W)
        mblock=torch.mul(mblock,masks_list)
        mblock=mblock.reshape(B*n,C,H,W)
        M_feature,_=self.backbone(mblock)
        M_feature=M_feature.reshape(B,n,self.feature_dim//2)
        R_feature=torch.cat((R_feature,M_feature),2)
        Data_list=[]
        for i in range(B):
            c_list=masks_class_list[i].squeeze().tolist()
            l_nodes=[]
            r_nodes=[]
            x=R_feature[i,-1,:].expand(V,self.feature_dim)*0
            for k in range(n):
                c=int(masks_class_list[i,k])
                if c==-1:
                    break
                x[c]=R_feature[i,k]
                x1=points_list[i,k,0]
                x2=points_list[i,k,1]
                y1=points_list[i,k,2]
                y2=points_list[i,k,3]
                center=(torch.div((x2-x1),2,rounding_mode='trunc')+x1, y1+torch.div((y2-y1),2,rounding_mode='trunc'))
                for c_r in edge_info[c]:
                    if c_r in c_list:
                        c_r_idx=c_list.index(c_r)
                        x1_r=points_list[i,c_r_idx,0]
                        x2_r=points_list[i,c_r_idx,1]
                        y1_r=points_list[i,c_r_idx,2]
                        y2_r=points_list[i,c_r_idx,3]
                        center_r=(torch.div((x2_r-x1_r),2,rounding_mode='trunc')+x1_r, y1_r+torch.div((y2_r-y1_r),2,rounding_mode='trunc'))
                        d=math.sqrt(abs(center[0]-center_r[0])*abs(center[0]-center_r[0])+abs(center_r[1]-center[1])*abs(center_r[1]-center[1]))
                        # if d<80 or Union((x1,y1,x2,y2),(x1_r,y1_r,x2_r,y2_r)):
                        if d<80:
                            l_nodes.append(c)
                            r_nodes.append(c_r)
            l_nodes=torch.tensor(l_nodes, dtype=torch.long)
            r_nodes=torch.tensor(r_nodes, dtype=torch.long)
            edge_idx=torch.stack((l_nodes,r_nodes),0).to(self.device)
            x=x.to(self.device)
            data = Data(x=x, edge_index=edge_idx)
            Data_list.append(data)
        loader = DataLoader(Data_list, batch_size=B,shuffle=False)
        for batch in loader:
            x=batch.x
            adj_t=batch.edge_index
        for i, layer in enumerate(self.layers[:-1]):
            new_x = layer(x, adj_t)
            new_x = self.non_linearity(new_x)
            # residual
            if i > 0 and self.use_residual:
                if self.use_residual_linear:
                    x = new_x + self.residuals[i](x)
                else:
                    x = new_x + x
                x = new_x + x
            else:
                x = new_x
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, adj_t)
        x=x.view(B,-1)
        x=self.fc2(x)
        if self.mode=='val':
            return x
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.interpolate(x, (H // 32, W // 32), mode='bilinear', align_corners=False)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x = F.interpolate(x, (H // 16, W // 16), mode='bilinear', align_corners=False)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x = F.interpolate(x, (H // 8, W // 8), mode='bilinear', align_corners=False)
        x=self.conv3(x)
        x=self.bn3(x)
        x=self.relu(x)
        x = F.interpolate(x, (H // 4, W // 4), mode='bilinear', align_corners=False)
        x=self.conv4(x)
        x=self.bn4(x)
        x=self.relu(x)
        x = F.interpolate(x, (H // 2, W // 2), mode='bilinear', align_corners=False)
        x=self.conv5(x)
        x=self.bn5(x)
        x=self.relu(x)
        x = F.interpolate(x, (H , W ), mode='bilinear', align_corners=False)
        x=self.conv6(x)
        return x

class ED_tf_predictor(nn.Module):
    def __init__(self,feature_dim=512,mode='train',device='cuda'):
        super(ED_tf_predictor, self).__init__()
        torch.cuda.manual_seed(233)
        self.device=device
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3],num_classes=feature_dim)
        self.tf = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8,batch_first=True)
        self.pe=PositionalEncoding(d_model=feature_dim,max_len=5)
        self.feature_dim=feature_dim
        self.mode=mode
        if mode=='train':
            self.conv1=nn.Conv2d(feature_dim, feature_dim//2, kernel_size=7, stride=1, padding=3)
            self.bn1 = nn.BatchNorm2d(feature_dim//2)
            self.relu = nn.ReLU(inplace=True)
            self.conv2=nn.Conv2d(feature_dim//2, feature_dim//4, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(feature_dim//4)
            self.conv3=nn.Conv2d(feature_dim//4, feature_dim//8, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(feature_dim//8)
            self.conv4=nn.Conv2d(feature_dim//8, feature_dim//16, kernel_size=3, stride=1, padding=1)
            self.bn4 = nn.BatchNorm2d(feature_dim//16)
            self.conv5=nn.Conv2d(feature_dim//16, feature_dim//32, kernel_size=3, stride=1, padding=1)
            self.bn5 = nn.BatchNorm2d(feature_dim//32)
            self.conv6=nn.Conv2d(feature_dim//32, 3, kernel_size=7, stride=1, padding=3)

    def forward(self, block):
        (B, N, C, SL, H, W) = block.shape
        block = block.transpose(2,3).reshape(B*N*SL, C, H, W)
        target_features=block.reshape(B,N*SL,C,H,W)[:,-1,:,:,:]
        feature,_ = self.backbone(block)
        if self.mode=='val':
            src_1=feature.reshape(B,N,SL,self.feature_dim)[:,0,:,:]
            pe_1=self.pe(src_1.transpose(0,1)).transpose(0,1)
            pred_features=self.tf(src_1+pe_1)     
            src_2=feature.reshape(B,N,SL,self.feature_dim)[:,1,:,:]
            pe_2=self.pe(src_2.transpose(0,1)).transpose(0,1)
            target_features=self.tf(src_2+pe_2)
            return pred_features, target_features
        feature=feature.reshape(B,N*SL,self.feature_dim)
        src=feature[:,:-1]
        pe=self.pe(src.transpose(0,1)).transpose(0,1)
        pred_features=self.tf(src+pe)
        pred_features = pred_features[:,-1,:].unsqueeze(-1).unsqueeze(-1)
        x = F.interpolate(pred_features, (H // 32, W // 32), mode='bilinear', align_corners=False)
        del pred_features
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x = F.interpolate(x, (H // 16, W // 16), mode='bilinear', align_corners=False)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x = F.interpolate(x, (H // 8, W // 8), mode='bilinear', align_corners=False)
        x=self.conv3(x)
        x=self.bn3(x)
        x=self.relu(x)
        x = F.interpolate(x, (H // 4, W // 4), mode='bilinear', align_corners=False)
        x=self.conv4(x)
        x=self.bn4(x)
        x=self.relu(x)
        x = F.interpolate(x, (H // 2, W // 2), mode='bilinear', align_corners=False)
        x=self.conv5(x)
        x=self.bn5(x)
        x=self.relu(x)
        x = F.interpolate(x, (H , W ), mode='bilinear', align_corners=False)
        x=self.conv6(x)
        return x,target_features