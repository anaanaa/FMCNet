# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

"""
Consider the width, length or depth of images is even number,
"""
import torch
from torch.autograd import Function


def Ein_m1(weights, inputt):
    return torch.einsum('nm,...mk->...nk', weights, inputt)


def Ein_m2(inputt, weights):
    return torch.einsum('...nk,km->...nm', inputt, weights)

class DWTFunction_3D(Function):

    @staticmethod
    def forward(ctx, input,
                matrix_Low_0, matrix_Low_1, matrix_Low_2,
                matrix_High_0, matrix_High_1, matrix_High_2):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_Low_2,
                              matrix_High_0, matrix_High_1, matrix_High_2)
        L = Ein_m1(matrix_Low_0, input)
        H = Ein_m1(matrix_High_0, input)
        LL = Ein_m2(L, matrix_Low_1).transpose(dim0=2, dim1=3)
        LH = Ein_m2(L, matrix_High_1).transpose(dim0=2, dim1=3)
        HL = Ein_m2(H, matrix_Low_1).transpose(dim0=2, dim1=3)
        HH = Ein_m2(H, matrix_High_1).transpose(dim0=2, dim1=3)
        LLL = Ein_m1(matrix_Low_2, LL).transpose(dim0=2, dim1=3)
        LLH = Ein_m1(matrix_Low_2, LH).transpose(dim0=2, dim1=3)
        LHL = Ein_m1(matrix_Low_2, HL).transpose(dim0=2, dim1=3)
        LHH = Ein_m1(matrix_Low_2, HH).transpose(dim0=2, dim1=3)
        HLL = Ein_m1(matrix_High_2, LL).transpose(dim0=2, dim1=3)
        HLH = Ein_m1(matrix_High_2, LH).transpose(dim0=2, dim1=3)
        HHL = Ein_m1(matrix_High_2, HL).transpose(dim0=2, dim1=3)
        HHH = Ein_m1(matrix_High_2, HH).transpose(dim0=2, dim1=3)
        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH

    @staticmethod
    def backward(ctx, grad_LLL, grad_LLH, grad_LHL, grad_LHH,
                 grad_HLL, grad_HLH, grad_HHL, grad_HHH):

        if grad_LLL.dtype == torch.float16 or grad_LLL.dtype == torch.bfloat16:
            grad_LLL = grad_LLL.type(torch.float32)
            grad_LLH = grad_LLH.type(torch.float32)
            grad_LHL = grad_LHL.type(torch.float32)
            grad_LHH = grad_LHH.type(torch.float32)
            grad_HLL = grad_HLL.type(torch.float32)
            grad_HLH = grad_HLH.type(torch.float32)
            grad_HHL = grad_HHL.type(torch.float32)
            grad_HHH = grad_HHH.type(torch.float32)

        matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2 = ctx.saved_variables
        grad_LL = torch.add(Ein_m1(matrix_Low_2.t(), grad_LLL.transpose(dim0=2, dim1=3)),
                            Ein_m1(matrix_High_2.t(), grad_HLL.transpose(dim0=2, dim1=3))).transpose(dim0=2,
                                                                                                           dim1=3)
        grad_LH = torch.add(Ein_m1(matrix_Low_2.t(), grad_LLH.transpose(dim0=2, dim1=3)),
                            Ein_m1(matrix_High_2.t(), grad_HLH.transpose(dim0=2, dim1=3))).transpose(dim0=2,
                                                                                                           dim1=3)
        grad_HL = torch.add(Ein_m1(matrix_Low_2.t(), grad_LHL.transpose(dim0=2, dim1=3)),
                            Ein_m1(matrix_High_2.t(), grad_HHL.transpose(dim0=2, dim1=3))).transpose(dim0=2,
                                                                                                           dim1=3)
        grad_HH = torch.add(Ein_m1(matrix_Low_2.t(), grad_LHH.transpose(dim0=2, dim1=3)),
                            Ein_m1(matrix_High_2.t(), grad_HHH.transpose(dim0=2, dim1=3))).transpose(dim0=2,
                                                                                                           dim1=3)
        grad_L = torch.add(Ein_m2(grad_LL, matrix_Low_1.t()), Ein_m2(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(Ein_m2(grad_HL, matrix_Low_1.t()), Ein_m2(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(Ein_m1(matrix_Low_0.t(), grad_L), Ein_m1(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None, None, None, None, None


class IDWTFunction_3D(Function):

    @staticmethod
    def forward(ctx, input_LLL, input_LLH, input_LHL, input_LHH,
                input_HLL, input_HLH, input_HHL, input_HHH,
                matrix_Low_0, matrix_Low_1, matrix_Low_2,
                matrix_High_0, matrix_High_1, matrix_High_2):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_Low_2,
                              matrix_High_0, matrix_High_1, matrix_High_2)

        input_LL = torch.add(Ein_m1(matrix_Low_2.t(), input_LLL.transpose(dim0=2, dim1=3)),
                             Ein_m1(matrix_High_2.t(), input_HLL.transpose(dim0=2, dim1=3))).transpose(dim0=2,
                                                                                                             dim1=3)
        input_LH = torch.add(Ein_m1(matrix_Low_2.t(), input_LLH.transpose(dim0=2, dim1=3)),
                             Ein_m1(matrix_High_2.t(), input_HLH.transpose(dim0=2, dim1=3))).transpose(dim0=2,
                                                                                                             dim1=3)
        input_HL = torch.add(Ein_m1(matrix_Low_2.t(), input_LHL.transpose(dim0=2, dim1=3)),
                             Ein_m1(matrix_High_2.t(), input_HHL.transpose(dim0=2, dim1=3))).transpose(dim0=2,
                                                                                                             dim1=3)
        input_HH = torch.add(Ein_m1(matrix_Low_2.t(), input_LHH.transpose(dim0=2, dim1=3)),
                             Ein_m1(matrix_High_2.t(), input_HHH.transpose(dim0=2, dim1=3))).transpose(dim0=2,
                                                                                                             dim1=3)
        input_L = torch.add(Ein_m2(input_LL, matrix_Low_1.t()), Ein_m2(input_LH, matrix_High_1.t()))
        input_H = torch.add(Ein_m2(input_HL, matrix_Low_1.t()), Ein_m2(input_HH, matrix_High_1.t()))
        output = torch.add(Ein_m1(matrix_Low_0.t(), input_L), Ein_m1(matrix_High_0.t(), input_H))
        return output

    @staticmethod
    def backward(ctx, grad_output):

        if grad_output.dtype == torch.float16 or grad_LLL.dtype == torch.bfloat16:
            grad_output = grad_output.type(torch.float32)

        matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2 = ctx.saved_variables
        grad_L = Ein_m1(matrix_Low_0, grad_output)
        grad_H = Ein_m1(matrix_High_0, grad_output)
        grad_LL = Ein_m2(grad_L, matrix_Low_1).transpose(dim0=2, dim1=3)
        grad_LH = Ein_m2(grad_L, matrix_High_1).transpose(dim0=2, dim1=3)
        grad_HL = Ein_m2(grad_H, matrix_Low_1).transpose(dim0=2, dim1=3)
        grad_HH = Ein_m2(grad_H, matrix_High_1).transpose(dim0=2, dim1=3)
        grad_LLL = Ein_m1(matrix_Low_2, grad_LL).transpose(dim0=2, dim1=3)
        grad_LLH = Ein_m1(matrix_Low_2, grad_LH).transpose(dim0=2, dim1=3)
        grad_LHL = Ein_m1(matrix_Low_2, grad_HL).transpose(dim0=2, dim1=3)
        grad_LHH = Ein_m1(matrix_Low_2, grad_HH).transpose(dim0=2, dim1=3)
        grad_HLL = Ein_m1(matrix_High_2, grad_LL).transpose(dim0=2, dim1=3)
        grad_HLH = Ein_m1(matrix_High_2, grad_LH).transpose(dim0=2, dim1=3)
        grad_HHL = Ein_m1(matrix_High_2, grad_HL).transpose(dim0=2, dim1=3)
        grad_HHH = Ein_m1(matrix_High_2, grad_HH).transpose(dim0=2, dim1=3)
        return grad_LLL, grad_LLH, grad_LHL, grad_LHH, grad_HLL, grad_HLH, grad_HHL, grad_HHH, None, None, None, None, None, None