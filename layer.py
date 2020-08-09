import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = 'cpu'

class ISubLayer(nn.Module):
    def __init__(self):
        super(ISubLayer, self).__init__()

    def forward(self, z, y, k, rho):
        '''
        z: from Z-sub problem [bs, ch, h, w]
        y: blurred image [bs, ch, h, w]
        k: kernel [bs, 1, h, w]
        '''
        bs, ch, h, w = z.shape
        # print("imageshape",z.shape)
        denominators = []
        upperlefts = []
        for i in range(bs):
            deno =[]
            upper =[]
            for c in range(3):
                V = self._t_psf2otf(k[i, 0, ...], [h, w])  # [1, h, w, 2]
                denominator = self._t_abs(V) ** 2  # [1, h, w]

                upperleft = self._t_multiply(self._t_conj(V),
                                             torch.rfft(y[i, c, ...].unsqueeze(0), signal_ndim=2, onesided=False)
                                             )  # [1, h, w, 2]
                deno.append(denominator)
                upper.append(upperleft)
            denominators.append(deno)
            upperlefts.append(upper)

        z_s = torch.zeros_like(y)
        for i in range(bs):
            for c in range(3):
                z_ = self._t_c2r_divide(upperlefts[i][c] + \
                                        rho * torch.rfft(z[i, c, ...].unsqueeze(0), signal_ndim=2, onesided=False)
                                        , denominators[i][c] + rho)  # [1, h, w, 2]
                z_ = torch.irfft(z_, signal_ndim=2, onesided=False)  # [1, h, w]
                z_s[i][c] = z_[0]
        # z_s = torch.stack(z_s, dim=0)
        # print("output shape",np.shape(z_s))
        return z_s

    def _t_psf2otf(self, psf, shape):
        '''
        otf: [1, h, w, 2]
        '''
        imshape_ = psf.shape
        shape = np.asarray(shape, dtype=int)
        imshape = np.asarray(imshape_, dtype=int)
        dshape = shape - imshape
        # padding
        idx, idy = np.indices(imshape)
        offx, offy = 0, 0
        pad_psf = torch.zeros(list(shape), dtype=psf.dtype).to(device)
        pad_psf[idx + offx, idy + offy] = psf
        for axis, axis_size in enumerate(imshape_):
            pad_psf = torch.roll(pad_psf, -int(axis_size / 2), dims=axis)
        otf = torch.rfft(pad_psf.unsqueeze(0), signal_ndim=2, onesided=False)
        return otf

    def _t_abs(self, input):
        '''
        @input: [bs, h, w, 2]
        @output: [bs, h, w]
        '''
        r, i = input[:, :, :, 0], input[:, :, :, 1]
        return (r ** 2 + i ** 2) ** 0.5

    def _t_conj(self, input):
        '''
        @input: [bs, h, w, 2]
        @output: [bs, h, w, 2]
        '''
        input_ = input.clone()
        input_[:, :, :, 1] = -input_[:, :, :, 1]
        return input_

    def _t_multiply(self, t1, t2):
        '''
        @input: [bs, h, w, 2]
        @output: [bs, h, w, 2]
        '''
        real1, imag1 = t1[:, :, :, 0], t1[:, :, :, 1]
        real2, imag2 = t2[:, :, :, 0], t2[:, :, :, 1]
        return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)

    def _t_c2r_divide(self, t1, t2):
        '''
        complex divided by real
        @input: [bs, h, w, 2], [bs, h, w]
        @output: [bs, h, w, 2]
        '''
        real1, imag1 = t1[..., 0], t1[..., 1]
        return torch.stack([real1 / t2, imag1 / t2], dim=-1)