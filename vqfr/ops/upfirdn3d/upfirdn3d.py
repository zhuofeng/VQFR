# modify from https://github.com/rosinality/stylegan2-pytorch/blob/master/op/upfirdn3d.py  # noqa:E501

import os
import torch
from torch.autograd import Function
from torch.nn import functional as F

BASICSR_JIT = os.getenv('BASICSR_JIT')
if BASICSR_JIT == 'True':
    from torch.utils.cpp_extension import load
    module_path = os.path.dirname(__file__)
    upfirdn3d_ext = load(
        'upfirdn3d',
        sources=[
            os.path.join(module_path, 'src', 'upfirdn3d.cpp'),
            os.path.join(module_path, 'src', 'upfirdn3d_kernel.cu'),
        ],
    )
else:
    try:
        from . import upfirdn3d_ext
    except ImportError:
        pass
        # avoid annoying print output
        # print(f'Cannot import deform_conv_ext. Error: {error}. You may need to: \n '
        #       '1. compile with BASICSR_EXT=True. or\n '
        #       '2. set BASICSR_JIT=True during running')


class upfirdn3dBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size):

        up_x, up_y, up_z = up
        down_x, down_y, down_z = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1, g_pad_z0, g_pad_z1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], out_size[2], 1)

        grad_input = upfirdn3d_ext.upfirdn3d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            down_z,
            up_x,
            up_y,
            up_z,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
            g_pad_z0,
            g_pad_z1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3], in_size[4])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.up_z = up_z
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.down_z = down_z
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.pad_z0 = pad_z0
        ctx.pad_z1 = pad_z1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], ctx.in_size[4], 1)

        gradgrad_out = upfirdn3d_ext.upfirdn3d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.up_z,
            ctx.down_x,
            ctx.down_y,
            ctx.down_z,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
            ctx.pad_z0,
            ctx.pad_z1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0],
        #                                  ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.in_size[1], ctx.in_size[2], ctx.out_size[0], ctx.out_size[1], ctx.out_size[2])

        return gradgrad_out, None, None, None, None, None, None, None, None, None, None, None, None


class upfirdn3d(Function):

    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y, up_z = up
        down_x, down_y, down_z = down
        pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1 = pad

        kernel_h, kernel_w, kernel_t = kernel.shape
        _, channel, in_h, in_w, in_t = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, in_t, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1, 2]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        out_t = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_z + 1
        ctx.out_size = (out_h, out_w, out_t)

        ctx.up = (up_x, up_y, up_z)
        ctx.down = (down_x, down_y, down_z)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_z0 = kernel_t - pad_z0 - 1

        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1
        g_pad_z1 = in_t * up_z - out_t * down_z + pad_z0 - up_z + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1, g_pad_z0, g_pad_z1)

        out = upfirdn3d_ext.upfirdn3d(input, kernel, up_x, up_y, up_z, down_x, down_y, down_z, pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1)
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w, out_t)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = upfirdn3dBackward.apply(
            grad_output,
            kernel,
            grad_kernel,
            ctx.up,
            ctx.down,
            ctx.pad,
            ctx.g_pad,
            ctx.in_size,
            ctx.out_size,
        )

        return grad_input, None, None, None, None


def upfirdn3d(input, kernel, up=1, down=1, pad=(0, 0, 0)):
    if input.device.type == 'cpu':
        out = upfirdn3d_native(input, kernel, up, up, up, down, down, down, pad[0], pad[1], pad[2], pad[0], pad[1], pad[2])
    else:
        out = upfirdn3d.apply(input, kernel, (up, up, up), (down, down, down), (pad[0], pad[1], pad[2], pad[0], pad[1], pad[2]))

    return out


def upfirdn3d_native(input, kernel, up_x, up_y, up_z, down_x, down_y, down_z, pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1):
    _, channel, in_h, in_w, in_t = input.shape
    input = input.reshape(-1, in_h, in_w, in_t, 1)

    _, in_h, in_w, in_t, minor = input.shape
    kernel_h, kernel_w, kernel_t = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, in_t, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0, 0, up_z - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, in_t * up_z, minor)

    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0), max(pad_z0, 0), max(pad_z1, 0)])
    out = out[:, max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), max(-pad_z0, 0):out.shape[2] - max(-pad_z1, 0), :, ]

    out = out.permute(0, 4, 1, 2, 3)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1, in_t * up_z + pad_z0 + pad_z1])
    w = torch.flip(kernel, [0, 1, 2]).view(1, 1, kernel_h, kernel_w, kernel_t)
    out = F.conv3d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
        in_t * up_z + pad_z0 + pad_z1 - kernel_t + 1,
    )
    out = out.permute(0, 2, 3, 4, 1)
    out = out[:, ::down_y, ::down_x, ::down_z, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    out_t = (in_t * up_z + pad_z0 + pad_z1 - kernel_t) // down_z + 1

    return out.view(-1, channel, out_h, out_w, out_t)
