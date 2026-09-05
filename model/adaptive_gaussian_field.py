import math

import torch
import torch.nn as nn


def _logit(probability):
    return math.log(probability / (1.0 - probability))


class AdaptiveGaussianField(nn.Module):
    """Content-adaptive Gaussian position field: MSG, MSG+AG, or core/context."""

    MODES = ('msg', 'msg_ag', 'full', 'g25_ag', 'g25_cc', 'g25_ag_cc')

    def __init__(self, mode='msg', sigma_bank=(12, 20, 25, 35, 50), base_sigma=25,
                 context_scale=2.0, gamma_init=0.05, beta_init=0.05):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError('unknown Gaussian-field mode: {}'.format(mode))
        sigma_bank = tuple(float(value) for value in sigma_bank)
        if base_sigma not in sigma_bank:
            raise ValueError('base_sigma must occur in sigma_bank')
        self.mode = mode
        self.register_buffer('sigma_bank', torch.tensor(sigma_bank, dtype=torch.float32))
        self.base_index = sigma_bank.index(float(base_sigma))
        self.base_sigma = float(base_sigma)
        self.context_scale = float(context_scale)
        self.a_max = 0.5 * math.log(3.0)

        self.local_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True),
        )
        uses_msg = mode in ('msg', 'msg_ag', 'full')
        uses_shape = mode in ('msg_ag', 'full', 'g25_ag', 'g25_ag_cc')
        uses_context = mode in ('full', 'g25_cc', 'g25_ag_cc')
        uses_gamma = mode in ('msg', 'msg_ag', 'full', 'g25_ag', 'g25_ag_cc')
        if uses_msg:
            self.alpha_mlp = nn.Sequential(nn.Linear(32, 32), nn.ReLU(inplace=True), nn.Linear(32, len(sigma_bank)))
            nn.init.zeros_(self.alpha_mlp[-1].weight)
            nn.init.zeros_(self.alpha_mlp[-1].bias)
            self.alpha_mlp[-1].bias.data[self.base_index] = 2.0
        if uses_gamma:
            self.gamma_mlp = nn.Sequential(nn.Linear(32, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))
            nn.init.zeros_(self.gamma_mlp[-1].weight)
            nn.init.constant_(self.gamma_mlp[-1].bias, _logit(gamma_init))
        if uses_shape:
            self.shape_mlp = nn.Sequential(nn.Linear(32, 16), nn.ReLU(inplace=True), nn.Linear(16, 2))
            nn.init.zeros_(self.shape_mlp[-1].weight)
            nn.init.zeros_(self.shape_mlp[-1].bias)
        if uses_context:
            self.beta_mlp = nn.Sequential(nn.Linear(32, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))
            nn.init.zeros_(self.beta_mlp[-1].weight)
            nn.init.constant_(self.beta_mlp[-1].bias, _logit(beta_init))

    @staticmethod
    def _grid(click_xy, height, width, device, dtype):
        batch = click_xy.shape[0]
        yy = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1)
        xx = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width)
        cx = click_xy[:, 0].to(device=device, dtype=dtype).view(batch, 1, 1, 1)
        cy = click_xy[:, 1].to(device=device, dtype=dtype).view(batch, 1, 1, 1)
        return xx - cx, yy - cy

    def build_isotropic_bank(self, click_xy, height, width, device, dtype):
        dx, dy = self._grid(click_xy, height, width, device, dtype)
        sigma = self.sigma_bank.to(device=device, dtype=dtype).view(1, -1, 1, 1)
        return torch.exp(-(dx.square() + dy.square()) / (2.0 * sigma.square()))

    def build_isotropic_single(self, click_xy, height, width, device, dtype, scale=1.0):
        dx, dy = self._grid(click_xy, height, width, device, dtype)
        sigma = torch.tensor(self.base_sigma * float(scale), device=device, dtype=dtype).view(1, 1, 1, 1)
        return torch.exp(-(dx.square() + dy.square()) / (2.0 * sigma.square()))

    def build_anisotropic_bank(self, click_xy, height, width, a, theta, device, dtype, scale=1.0):
        dx, dy = self._grid(click_xy, height, width, device, dtype)
        cosine, sine = torch.cos(theta).view(-1, 1, 1, 1), torch.sin(theta).view(-1, 1, 1, 1)
        u, v = cosine * dx + sine * dy, -sine * dx + cosine * dy
        sigma = self.sigma_bank.to(device=device, dtype=dtype).view(1, -1, 1, 1)
        a = a.to(dtype=dtype).view(-1, 1, 1, 1)
        sigma_x, sigma_y = scale * sigma * torch.exp(a), scale * sigma * torch.exp(-a)
        return torch.exp(-0.5 * (u.square() / sigma_x.square() + v.square() / sigma_y.square()))

    def build_anisotropic_single(self, click_xy, height, width, a, theta, device, dtype, scale=1.0):
        dx, dy = self._grid(click_xy, height, width, device, dtype)
        cosine, sine = torch.cos(theta).view(-1, 1, 1, 1), torch.sin(theta).view(-1, 1, 1, 1)
        u, v = cosine * dx + sine * dy, -sine * dx + cosine * dy
        sigma = torch.tensor(self.base_sigma * float(scale), device=device, dtype=dtype).view(1, 1, 1, 1)
        a = a.to(dtype=dtype).view(-1, 1, 1, 1)
        sigma_x, sigma_y = sigma * torch.exp(a), sigma * torch.exp(-a)
        return torch.exp(-0.5 * (u.square() / sigma_x.square() + v.square() / sigma_y.square()))

    def _shape(self, z):
        shape = self.shape_mlp(z)
        return self.a_max * torch.tanh(shape[:, 0]), 0.5 * math.pi * torch.tanh(shape[:, 1])

    def forward(self, rgb, click_xy, return_aux=False):
        height, width = rgb.shape[-2:]
        base = self.build_isotropic_single(click_xy, height, width, rgb.device, rgb.dtype)
        feature = self.local_encoder(rgb)
        z = (feature * base).sum(dim=(2, 3)) / (base.sum(dim=(2, 3)) + 1e-6)

        if self.mode == 'g25_cc':
            wide = self.build_isotropic_single(click_xy, height, width, rgb.device, rgb.dtype, self.context_scale)
            context = torch.clamp(wide - base, min=0.0)
            beta = torch.sigmoid(self.beta_mlp(z)).view(-1, 1, 1, 1)
            position = torch.clamp(base + beta * context, 0.0, 1.0)
            aux = {'beta': beta.squeeze(-1).squeeze(-1).squeeze(-1), 'base': base, 'context': context}
            return (position, aux) if return_aux else position

        if self.mode in ('g25_ag', 'g25_ag_cc'):
            a, theta = self._shape(z)
            core = self.build_anisotropic_single(click_xy, height, width, a, theta, rgb.device, rgb.dtype)
            gamma = torch.sigmoid(self.gamma_mlp(z)).view(-1, 1, 1, 1)
            if self.mode == 'g25_ag':
                position = (1.0 - gamma) * base + gamma * core
                aux = {'gamma': gamma.squeeze(-1).squeeze(-1).squeeze(-1), 'a': a, 'theta': theta, 'base': base, 'core': core}
            else:
                wide = self.build_anisotropic_single(click_xy, height, width, a, theta, rgb.device, rgb.dtype,
                                                     self.context_scale)
                context = torch.clamp(wide - core, min=0.0)
                beta = torch.sigmoid(self.beta_mlp(z)).view(-1, 1, 1, 1)
                core_context = torch.clamp(core + beta * context, 0.0, 1.0)
                position = (1.0 - gamma) * base + gamma * core_context
                aux = {'gamma': gamma.squeeze(-1).squeeze(-1).squeeze(-1), 'beta': beta.squeeze(-1).squeeze(-1).squeeze(-1),
                       'a': a, 'theta': theta, 'base': base, 'core': core, 'context': context}
            return (position, aux) if return_aux else position

        isotropic = self.build_isotropic_bank(click_xy, height, width, rgb.device, rgb.dtype)
        alpha = torch.softmax(self.alpha_mlp(z), dim=1)
        gamma = torch.sigmoid(self.gamma_mlp(z)).view(-1, 1, 1, 1)

        if self.mode == 'msg':
            core = (alpha[:, :, None, None] * isotropic).sum(dim=1, keepdim=True)
            position = (1.0 - gamma) * base + gamma * core
            aux = {'alpha': alpha, 'gamma': gamma.squeeze(-1).squeeze(-1).squeeze(-1), 'base': base, 'core': core}
        else:
            a, theta = self._shape(z)
            anisotropic = self.build_anisotropic_bank(click_xy, height, width, a, theta, rgb.device, rgb.dtype)
            core = (alpha[:, :, None, None] * anisotropic).sum(dim=1, keepdim=True)
            if self.mode == 'msg_ag':
                position = (1.0 - gamma) * base + gamma * core
                aux = {'alpha': alpha, 'gamma': gamma.squeeze(-1).squeeze(-1).squeeze(-1), 'a': a, 'theta': theta, 'base': base, 'core': core}
            else:
                wide_bank = self.build_anisotropic_bank(click_xy, height, width, a, theta, rgb.device, rgb.dtype,
                                                        scale=self.context_scale)
                wide = (alpha[:, :, None, None] * wide_bank).sum(dim=1, keepdim=True)
                context = torch.clamp(wide - core, min=0.0)
                beta = torch.sigmoid(self.beta_mlp(z)).view(-1, 1, 1, 1)
                core_context = torch.clamp(core + beta * context, 0.0, 1.0)
                position = (1.0 - gamma) * base + gamma * core_context
                aux = {'alpha': alpha, 'gamma': gamma.squeeze(-1).squeeze(-1).squeeze(-1), 'beta': beta.squeeze(-1).squeeze(-1).squeeze(-1),
                       'a': a, 'theta': theta, 'base': base, 'core': core, 'context': context}
        return (position, aux) if return_aux else position
