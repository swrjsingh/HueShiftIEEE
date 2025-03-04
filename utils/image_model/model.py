import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NoiseScheduler:
  def __init__(self,beta_0=0.0001,beta_T=0.02,T=1000):
    self.beta_0=beta_0
    self.beta_T=beta_T
    self.T=T
    self.sqrt_alpha_bar=[1]
    self.sqrt_one_minus_alpha_bar=[0]
    self.alpha_bar=[1]
    self.alpha=[1]

    alpha_bar=1
    for i in range(0,T):
      beta=self.beta_0+(self.beta_T-self.beta_0)*i/T
      alpha=1-beta
      alpha_bar*=alpha
      self.alpha.append(alpha)
      self.alpha_bar.append(alpha_bar)
      self.sqrt_alpha_bar.append(alpha_bar**0.5)
      self.sqrt_one_minus_alpha_bar.append((1-alpha_bar)**0.5)

    self.sqrt_alpha_bar=torch.Tensor(self.sqrt_alpha_bar).to(device)
    self.sqrt_one_minus_alpha_bar=torch.Tensor(self.sqrt_one_minus_alpha_bar).to(device)
    self.alpha_bar=torch.Tensor(self.alpha_bar).to(device)
    self.alpha=torch.Tensor(self.alpha).to(device)

  def get_timestep(self,x_0,t):
    sqrt_alpha_bar_t=self.sqrt_alpha_bar[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    sqrt_one_minus_alpha_bar_t=self.sqrt_one_minus_alpha_bar[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)

    epsilon=torch.randn_like(x_0[:,1:3,:,:])

    x_t=x_0.clone()
    x_t[:,1:3,:,:]=sqrt_alpha_bar_t*x_t[:,1:3,:,:] + sqrt_one_minus_alpha_bar_t*epsilon

    return x_t,epsilon

  def sample_X_0(self,X_t,epsilon,t):
    sqrt_alpha_bar_t=self.sqrt_alpha_bar[t]
    sqrt_one_minus_alpha_bar_t=self.sqrt_one_minus_alpha_bar[t]
    X_0=(X_t-sqrt_one_minus_alpha_bar_t*epsilon)/sqrt_alpha_bar_t
    X_0=torch.clamp(X_0,-1,1)
    return X_0

  def sample_previous_timestep(self,X_t,epsilon,t):
    sqrt_alpha_bar_t=self.sqrt_alpha_bar[t]
    sqrt_one_minus_alpha_bar_t=self.sqrt_one_minus_alpha_bar[t]

    alpha=self.alpha[t]
    beta=1-alpha

    ab_t1=(X_t[:,1:3,:,:]-(beta*epsilon)/sqrt_one_minus_alpha_bar_t)/(alpha**0.5)

    if t!=1:
      var=beta*(1-self.alpha_bar[t-1])/(1-self.alpha_bar[t])
      z=torch.randn_like(ab_t1).to(device)
      ab_t1+=z*(var**0.5)

    X_t[:,1:3,:,:]=ab_t1


    return X_t

class TimeEmbedding(nn.Module):
  def __init__(self,latent_dim,embedding_n=1000):
    super().__init__()

    self.latent_dim=latent_dim
    self.embedding_n=embedding_n

    self.embed_silu=nn.SiLU()
    # self.embed_fc=nn.Linear(in_features=self.latent_dim,out_features=64*8*8)
    self.embed_latent=nn.Linear(in_features=self.latent_dim,out_features=self.latent_dim)

  def forward(self,t):
    t=t.to(device).unsqueeze(1).float()
    i=torch.arange(0,self.latent_dim,2).to(device)
    div_term=torch.exp(-2*i/self.latent_dim*math.log(self.embedding_n)).unsqueeze(0)
    t_encoding=torch.zeros((t.shape[0],self.latent_dim)).to(device)
    t_encoding[:,0::2]=torch.sin(torch.matmul(t,div_term)).to(device)
    t_encoding[:,1::2]=torch.cos(torch.matmul(t,div_term)).to(device)

    t_latent_embedding=self.embed_latent(t_encoding)
    t_latent_embedding=self.embed_silu(t_latent_embedding)
    return t_latent_embedding

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, C, H*W)
        k = k.view(B, C, H*W)
        v = v.view(B, C, H*W)

        attn = torch.einsum('bci,bcj->bij', q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=2)

        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view(B, C, H, W)
        return x + self.proj(out)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, use_attention=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        self.attention = SelfAttention(out_channels) if use_attention else nn.Identity()
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        h = self.attention(h)
        return h + self.residual(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, time_dim=256, base_channels=128,mock=False):
        super().__init__()
        self.mock=mock
        self.time_embedding = TimeEmbedding(time_dim)
        # self.time_mlp = nn.Sequential(
        #     nn.Linear(time_dim, time_dim * 4),
        #     nn.SiLU(),
        #     nn.Linear(time_dim * 4, time_dim)
        # )

        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsample blocks
        self.down1 = nn.ModuleList([
            ResBlock(base_channels, base_channels, time_dim),
            ResBlock(base_channels, base_channels, time_dim),
            nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1)
        ])
        # 1,16,16

        self.down2 = nn.ModuleList([
            ResBlock(base_channels, base_channels*2, time_dim),
            ResBlock(base_channels*2, base_channels*2, time_dim, use_attention=True),
            nn.Conv2d(base_channels*2, base_channels*2, 4, stride=2, padding=1)
        ])
        # 2,8,8

        self.down3 = nn.ModuleList([
            ResBlock(base_channels*2, base_channels*4, time_dim),
            ResBlock(base_channels*4, base_channels*4, time_dim, use_attention=True),
            nn.Conv2d(base_channels*4, base_channels*4, 4, stride=2, padding=1)
        ])
        # 4,4,4

        # Middle blocks
        self.middle = nn.ModuleList([
            ResBlock(base_channels*4, base_channels*4, time_dim, use_attention=True),
            ResBlock(base_channels*4, base_channels*4, time_dim, use_attention=True),
            ResBlock(base_channels*4, base_channels*4, time_dim, use_attention=True)
        ])
        # 4,4,4

        # Upsample blocks
        self.up1 = nn.ModuleList([
            ResBlock(base_channels*8, base_channels*2, time_dim),
            ResBlock(base_channels*2, base_channels*2, time_dim, use_attention=True),
            nn.ConvTranspose2d(base_channels*2, base_channels*2, 4, stride=2, padding=1)
        ])
        # 2,8,8

        self.up2 = nn.ModuleList([
            ResBlock(base_channels*4, base_channels*1, time_dim),
            ResBlock(base_channels*1, base_channels*1, time_dim, use_attention=True),
            nn.ConvTranspose2d(base_channels*1, base_channels*1, 4, stride=2, padding=1)
        ])
        # 1,16,16

        self.up3 = nn.ModuleList([
            ResBlock(base_channels*2, base_channels, time_dim),
            ResBlock(base_channels, base_channels, time_dim),
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)
        ])
        # 1,32,32

        # Output
        self.out = nn.Sequential(
            # nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
      # Time embedding
      if not self.mock:
        t = self.time_embedding(t)
        # t = self.time_mlp(t)
      else:
        t=torch.zeros(x.shape[0],256).to(device)

      # Initial conv
      x = self.init_conv(x)
      # print("After init_conv:", x.shape)
      skips = [x]

      # Downsample
      for i, layer in enumerate(self.down1):
          x = layer(x, t) if isinstance(layer, ResBlock) else layer(x)
          # print(f"Down1 layer {i}:", x.shape)
      skips.append(x)

      for i, layer in enumerate(self.down2):
          x = layer(x, t) if isinstance(layer, ResBlock) else layer(x)
          # print(f"Down2 layer {i}:", x.shape)
      skips.append(x)

      for i, layer in enumerate(self.down3):
          x = layer(x, t) if isinstance(layer, ResBlock) else layer(x)
          # print(f"Down3 layer {i}:", x.shape)
      skips.append(x)

      # Middle
      for i, layer in enumerate(self.middle):
          x = layer(x, t)
          # print(f"Middle layer {i}:", x.shape)

      # # Upsample
      # print("\nSkip connection shapes:", [s.shape for s in skips])

      first=True
      skip=skips.pop()
      for layer in self.up1:
          # print(x.shape)
          # print()

          if isinstance(layer, ResBlock):
            if first:
              x=torch.cat([x,skip],dim=1)
              first=False
            x = layer(x, t)
          else:
              x = layer(x)
      # print(x.shape)

      first=True
      skip=skips.pop()
      for layer in self.up2:


          if isinstance(layer, ResBlock):
            if first:
              x=torch.cat([x,skip],dim=1)
              first=False
            x = layer(x, t)
          else:
              x = layer(x)
      # print(x.shape)

      first=True
      skip=skips.pop()
      for layer in self.up3:
          # print("\nSkip connection shapes:", [s.shape for s in skips])
          # print(x.shape)
          # print()

          if isinstance(layer, ResBlock):
            if first:
              x=torch.cat([x,skip],dim=1)
              first=False
            x = layer(x, t)
          else:
              x = layer(x)
      # print(x.shape)

      return self.out(x)

    def sample(self,X_T,ns):
      X_t=X_T
      for t in range(ns.T,0,-1):
        with torch.no_grad():
          epsilon=self.forward(X_t,torch.Tensor([t]).to(device))
        X_t=ns.sample_previous_timestep(X_t,epsilon,t)
      return X_t



def run_image_model(image):
    model = UNet(
        in_channels=3,  # LAB space input
        out_channels=2, # Only predict noise for a,b channels
        time_dim=256,
        base_channels=64,
        mock=False,
    )
    model.to(device)
    model.load_state_dict(torch.load("utils/image_model/model.pth", map_location=device))
    ns=NoiseScheduler()
    return model.sample(image,ns)