import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from models.models_utils import weights_init

try:
    from timm.models.swin_transformer import SwinTransformerBlock
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 1,
                    stride =stride, padding=0,bias=False)

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
        stride =stride, padding=1,bias=False)

class irnn_layer(nn.Module):
    def __init__(self,in_channels):
        super(irnn_layer,self).__init__()
        self.left_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.right_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.up_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.down_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        
    def forward(self,x):
        _,_,H,W = x.shape
        top_left = x.clone()
        top_right = x.clone()
        top_up = x.clone()
        top_down = x.clone()
        top_left[:,:,:,1:] = F.relu(self.left_weight(x)[:,:,:,:W-1]+x[:,:,:,1:],inplace=False)
        top_right[:,:,:,:-1] = F.relu(self.right_weight(x)[:,:,:,1:]+x[:,:,:,:W-1],inplace=False)
        top_up[:,:,1:,:] = F.relu(self.up_weight(x)[:,:,:H-1,:]+x[:,:,1:,:],inplace=False)
        top_down[:,:,:-1,:] = F.relu(self.down_weight(x)[:,:,1:,:]+x[:,:,:H-1,:],inplace=False)
        return (top_up,top_right,top_down,top_left)

class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention,self).__init__()
        self.out_channels = int(in_channels/2)
        self.conv1 = nn.Conv2d(in_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels,4,kernel_size=1,padding=0,stride=1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out

class SAM(nn.Module):
    def __init__(self,in_channels,out_channels,attention=1):
        super(SAM,self).__init__()
        self.out_channels = out_channels
        self.irnn1 = irnn_layer(self.out_channels)
        self.irnn2 = irnn_layer(self.out_channels)
        self.conv_in = conv3x3(in_channels,self.out_channels)
        self.relu1 = nn.ReLU(True)
        self.conv1 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(self.out_channels*4,self.out_channels,kernel_size=1,stride=1,padding=0)
        self.conv3 = nn.Conv2d(self.out_channels*4,self.out_channels,kernel_size=1,stride=1,padding=0)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels,1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv1(x)
        top_up,top_right,top_down,top_left = self.irnn1(out)
        if self.attention:
            top_up = top_up * weight[:,0:1,:,:]
            top_right = top_right * weight[:,1:2,:,:]
            top_down = top_down * weight[:,2:3,:,:]
            top_left = top_left * weight[:,3:4,:,:]
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv2(out)
        top_up,top_right,top_down,top_left = self.irnn2(out)
        if self.attention:
            top_up = top_up * weight[:,0:1,:,:]
            top_right = top_right * weight[:,1:2,:,:]
            top_down = top_down * weight[:,2:3,:,:]
            top_left = top_left * weight[:,3:4,:,:]
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        mask = self.sigmod(self.conv_out(out))
        return mask

class BasicWindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    def forward(self, x):
        B, C, H, W = x.shape
        x_res = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape
        x = x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size * self.window_size, C)
        qkv = self.qkv(x).reshape(-1, self.window_size * self.window_size, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * ((C // self.num_heads) ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        x = self.proj(x)
        x = x.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        x = x[:, :H, :W, :].permute(0, 3, 1, 2)
        x = x + x_res
        x_res = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        x = x + x_res
        return x

class SwinBottleneck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if HAS_TIMM:
            self.block1 = SwinTransformerBlock(dim=dim, input_resolution=(16, 16), num_heads=8, window_size=8, shift_size=0)
            self.block2 = SwinTransformerBlock(dim=dim, input_resolution=(16, 16), num_heads=8, window_size=8, shift_size=4)
            self.use_timm = True
        else:
            self.block1 = BasicWindowAttention(dim, 8, 8)
            self.block2 = BasicWindowAttention(dim, 8, 8)
            self.use_timm = False

    def forward(self, x):
        if self.use_timm:
            B, C, H, W = x.shape
            x_hwc = x.permute(0, 2, 3, 1)
            try:
                x_out = self.block1(x_hwc)
                x_out = self.block2(x_out)
            except TypeError:
                try:
                    x_L = x.flatten(2).transpose(1, 2)
                    x_L = self.block1(x_L, input_resolution=(H, W))
                    x_L = self.block2(x_L, input_resolution=(H, W))
                    x_out = x_L.view(B, H, W, C)
                except Exception:
                    x_out = x_hwc
            return x_out.permute(0, 3, 1, 2)
        else:
            return self.block2(self.block1(x))

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(out_c)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.1) if dropout else nn.Identity()
    def forward(self, x, skip_x, sam_module):
        attn = sam_module(skip_x)
        skip_x = skip_x * attn
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = torch.cat([x, skip_x], dim=1)
        return x, attn

class SPANet(nn.Module):
    # ← CHANGED: out_channels default is now 4 to match trained checkpoint
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 512)
        self.bottleneck = SwinBottleneck(512)
        self.sam4 = SAM(512, 512)
        self.up4 = UpBlock(512, 512, dropout=True)
        self.sam3 = SAM(256, 256)
        self.up3 = UpBlock(1024, 256, dropout=True)
        self.sam2 = SAM(128, 128)
        self.up2 = UpBlock(512, 128, dropout=False)
        self.sam1 = SAM(64, 64)
        self.up1 = UpBlock(256, 64, dropout=False)
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d0 = self.inc(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        b = self.bottleneck(d4)
        u4, att4 = self.up4(b, d3, self.sam4)
        u3, att3 = self.up3(u4, d2, self.sam3)
        u2, att2 = self.up2(u3, d1, self.sam2)
        u1, att1 = self.up1(u2, d0, self.sam1)
        out = self.final_conv(u1)
        return att1, out

class Generator(nn.Module):
    # ← CHANGED: out_ch default is now 4 to match trained checkpoint
    def __init__(self, gpu_ids, in_ch=4, out_ch=4):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.gen = nn.Sequential(OrderedDict([('gen', SPANet(in_channels=in_ch, out_channels=out_ch))]))
        self.gen.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids:
            return nn.parallel.data_parallel(self.gen, x, self.gpu_ids)
        else:
            return self.gen(x)