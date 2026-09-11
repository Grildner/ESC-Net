class E(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(E, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.norm = nn.GroupNorm(1, mip)
        self.act = StarReLU() # nn.SiLU
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.guide_conv = nn.Conv2d(oup,1,1)
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        att = a_h * a_w
        out = identity * att + 0.1 * identity
        guide = self.guide_conv(att)
        return out, guide
        
class S(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)

        if dyscope:
            self.scope = nn.Sequential(
                nn.Conv2d(in_channels, max(in_channels//4, 8), 1),  # nonlinear enhance
                nn.ReLU(inplace=True),
                nn.Conv2d(max(in_channels//4, 8), out_channels, 1, bias=False)
            )
            # better init
            for m in self.scope:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        else:
            self.scope = None

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope') and self.scope is not None:
            scope_factor = self.scope(x).sigmoid() * 0.5
            offset = self.offset(x) * scope_factor + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope') and self.scope is not None:
            scope_factor = F.pixel_unshuffle(self.scope(x_).sigmoid(), self.scale) * 0.5
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * scope_factor + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self,x,context=None):
        if context is not None:
            context=F.interpolate(
                context,
                x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            x=x*(1+0.2*context)

        if self.style=='pl':
            return self.forward_pl(x)

        return self.forward_lp(x)
    
class C(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1,
                 shortcut: bool = True, e: float = 0.5,
                 num_param: int = 5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU(inplace=True),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * self.c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, num_param=num_param,
                         shortcut=shortcut, e=1.0)
            for _ in range(n)
        )

    def forward(self,x,context=None):
        y=self.cv1(x)
        if context is not None:
            context=F.interpolate(
                context,
                y.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            y=y*(1+0.1*context)
        z=list(y.chunk(2,dim=1))
        z.extend(
            m(z[-1]) for m in self.m
        )
        return self.cv2(torch.cat(z,1))