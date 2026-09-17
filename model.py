"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    # if T <= 1:
    #     return torch.tensor([beta_start])

    # betas = []
    # each_beta = (beta_end - beta_start) / (T - 1)
    # for t in range(T):
    #     betas.append(beta_start)
    #     beta_start += each_beta
    # betas = torch.tensor(betas)
    # return betas 
    return torch.linspace(beta_start,beta_end,T)

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    # TODO: return 1 - betas
    return 1 - betas

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    # TODO: cumulative product of alphas
    return torch.cumprod(alphas,dim=0)

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    timestep = a[t]
    timestep = timestep.reshape(-1,1,1,1)
    return timestep

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    # TODO: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise
    t = t.reshape(-1,1,1,1)
    return torch.sqrt(alphas_cumprod[t]) * x0 + torch.sqrt(1 - alphas_cumprod[t]) * noise

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    # TODO: build betas, alphas, alphas_cumprod and useful sqrts
    # betas = []
    # each_beta = (beta_end - beta_start) / (T - 1)
    # add_t = beta_start
    # for t in range(T):
    #     betas.append(add_t)
    #     add_t += each_beta
    betas = linear_beta_schedule(T,beta_start,beta_end)

    alphas = 1 - betas 

    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    return {
        "T": T,
        "alphas": alphas,
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod
    }

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    # TODO: MSE between predicted and true noise
    # return ((noise - noise_pred) ** 2).mean()
    return F.mse_loss(noise_pred,noise)

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    # TODO: q_sample -> model -> MSE(noise_pred, noise)
    t = t.reshape(-1,1,1,1)
    # xt = alphas_cumprod[t].sqrt() * x0 + (1 - alphas_cumprod[t]).sqrt() * noise
    xt = q_sample(x0,t,noise,alphas_cumprod)
    predict_noise = model(xt,t)
    # 预测目标是 原始 noise 而不是 真正加上去的 noise
    return F.mse_loss(predict_noise, noise)

# Step 9 - timestep_embedding
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    # TODO: sinusoidal timestep embedding of shape (B, dim)
    half = int(dim) // 2

    i = torch.arange(half) # 0 ... h - 1， dim=h

    exponent = i / max(half - 1, 1)

    w_i = (1 / (10000 ** exponent)) # dim
    # timesteps.cat(torch.sin(w_i * t))
    # timesteps.cat(torch.cos(w_i * t))
    # t , dim=B

    w_i = w_i.reshape(1,-1) # (1, h)
    t = t.reshape(-1,1) # (B, 1)

    return torch.cat([torch.sin(w_i * t),torch.cos(w_i * t)],dim=1)

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    # TODO: initialize tiny residual denoiser parameters
    torch.manual_seed(seed)

    params = {
        "conv_in_w":  (0.02 * torch.randn(hidden, in_ch, 3, 3)).requires_grad_(),
        "conv_in_b":  torch.zeros(hidden, requires_grad=True),

        "time_mlp_w": (0.02 * torch.randn(hidden, time_dim)).requires_grad_(),
        "time_mlp_b": torch.zeros(hidden, requires_grad=True),

        "conv_mid_w": (0.02 * torch.randn(hidden, hidden, 3, 3)).requires_grad_(),
        "conv_mid_b": torch.zeros(hidden, requires_grad=True),

        "conv_out_w": (0.02 * torch.randn(in_ch, hidden, 3, 3)).requires_grad_(),
        "conv_out_b": torch.zeros(in_ch, requires_grad=True),
    }

    return params

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
    # TODO: time-conditioned tiny CNN predicting noise
    h = F.conv2d(x,params['conv_in_w'],params['conv_in_b'],padding=1)
    temb = timestep_embedding(t, params['time_mlp_w'].shape[1])
    temb = F.relu(F.linear(temb,params['time_mlp_w'],params['time_mlp_b']))
    h += temb[:,:,None,None]
    h = F.relu(h)
    return F.conv2d(h,params['conv_out_w'],params['conv_out_b'],padding=1)

# Step 12 - make_blob_dataset
import torch
import torch.nn.functional as F

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    # TODO: n images with a random bright disk on a black background
    torch.manual_seed(seed)

    # 1. 创建 n 张全黑图片
    x = torch.zeros((n, 1, size, size), dtype=torch.float32)

    # 2. 圆的半径
    radius = size // 4

    # 3. 每个像素的坐标
    yy, xx = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        indexing="ij"
    )

    # 4. 每张图片随机放一个圆
    for i in range(n):

        # 随机生成圆心
        cy, cx = torch.randint(
            radius,
            size - radius,
            (2,)
        )

        # 判断哪些像素在圆里面
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2

        # 圆内部设为白色
        x[i, 0][mask] = 1.0

    return x

# Step 13 - ddpm_train_step
import torch
import torch.nn.functional as F

def ddpm_train_step(params: dict, x0, schedule: dict, lr: float = 1e-2, seed: int = 0) -> tuple[dict, float]:
    # TODO: sample t,noise -> loss -> SGD on params
    torch.manual_seed(seed)
    t = torch.randint(schedule['T'],(x0.shape[0],)) # sample t
    t_reshape = t.reshape(-1,1,1,1)
    # sample noise
    noise = torch.randn_like(x0)

    x_t = schedule["sqrt_alphas_cumprod"][t_reshape] * x0 + schedule["sqrt_one_minus_alphas_cumprod"][t_reshape] * noise

    pred_noise = tiny_unet_forward(x_t,t,params)

    loss =  F.mse_loss(pred_noise, noise)

    def sgd(loss,ps,lr):
        loss.backward()

        new_p = {}

        for name, p in ps.items():
            if p.grad is not None:
                p_new = (p - lr * p.grad).detach().requires_grad_(True)
            else:
                p_new = p.clone().detach().requires_grad_(True)

            new_p[name] = p_new
        return new_p
    return sgd(loss,params,lr), float(loss)

# Step 14 - train_ddpm
import torch
import torch.nn.functional as F

def train_ddpm(dataset, params: dict, schedule: dict, num_steps: int = 50, batch_size: int = 16, lr: float = 1e-2, seed: int = 0) -> tuple[dict, list]:
    # TODO: minibatch SGD training loop
    g = torch.Generator()
    history = []
    for step in range(num_steps):
        # set the seed as seed + current step
        current_seed = seed + step
        g.manual_seed(current_seed) 
        minibatch = torch.randint(dataset.size()[0],(batch_size,))
        x0 = dataset[minibatch]
        params, loss = ddpm_train_step(params,x0,schedule,lr,seed)
        history.append(loss)
    return params, history

# Step 15 - predict_x0_from_eps
import torch
import torch.nn.functional as F

def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    # TODO: invert the q_sample equation for x0
    a = extract_into_batch(alphas_cumprod,t,x_t) # get schedule data with timesteps
    x0_hat = (x_t - torch.sqrt(1 - a) * eps) / torch.sqrt(a)
    return x0_hat

# Step 16 - ddpm_p_mean_variance
import torch
import torch.nn.functional as F

def ddpm_p_mean_variance(x_t, t, eps, schedule: dict):
    # TODO: return (posterior_mean, variance, x0_hat)
    x0_hat = predict_x0_from_eps(x_t, t, eps, schedule["alphas_cumprod"]).clamp(-1,1)
    # if t == 0:

    alphas_cumprod_t_minus_1 = torch.cat([torch.ones_like(schedule["alphas_cumprod"][:1]),schedule["alphas_cumprod"][:-1]])[t]

    mu = (
            torch.sqrt(alphas_cumprod_t_minus_1)
            * schedule["betas"][t] 
            / (1 - schedule["alphas_cumprod"][t])
        ).reshape(-1,1,1,1) * x0_hat + (
                torch.sqrt(schedule["alphas"][t]) * 
                (1 - alphas_cumprod_t_minus_1) 
                / (1 - schedule["alphas_cumprod"][t])
            ).reshape(-1,1,1,1) * x_t
    return (mu, schedule["betas"][t].reshape(-1,1,1,1), x0_hat)

# Step 17 - ddpm_p_sample
import torch
import torch.nn.functional as F

def ddpm_p_sample(x_t, t, params: dict, schedule: dict, noise=None):
    # TODO: one reverse step x_t -> x_{t-1}
    if noise is None:
        noise = torch.randn_like(x_t)
    pred_noise = tiny_unet_forward(x_t, t, params)

    # calculate the reverse gaussian from predicted noise
    mean, var, _ = ddpm_p_mean_variance(x_t, t, pred_noise, schedule) # x_t-1 就是直接从这个 gaussian 中采样得到了

    # torch tensor 的二元条件项应该通过 mask 实现
    no_zero_mask_t = (t != 0).reshape(-1,1,1,1)

    x_prev = mean + no_zero_mask_t * torch.sqrt(var) * noise
    return x_prev

# Step 18 - ddpm_sample_loop
import torch
import torch.nn.functional as F

def ddpm_sample_loop(params: dict, schedule: dict, shape: tuple, seed: int = 0):
    # TODO: ancestral sampling from pure noise to x0
    torch.manual_seed(seed) 
    x = torch.randn(shape)
    T = schedule["T"]
    for t in range(0,T):
        t = torch.full_like(torch.tensor(x.shape[0],),fill_value=t)
        x = ddpm_p_sample(x,t,params,schedule)
    return x

# Step 19 - sample_quality_mse
import torch
import torch.nn.functional as F

def sample_quality_mse(samples, dataset) -> float:
    # TODO: mean over samples of min MSE to any dataset image
    diff = samples[:, None] - dataset[None, :]
    mse = (diff ** 2).mean(dim=(2, 3, 4))
    min_mse = mse.min(dim=1).values
    return min_mse.mean().item()

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

