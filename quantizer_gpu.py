'''
File to quantize activations

input - activation map after RELU
ifTraining - if True Just approximate activation and train weights for the next layer, otherwise use precomputed value from the table
weights - weights of the next layer
tree - look-up table for quantization during training
    TODO !!! Can be substituted by any other quantization (clustering method)
lookup_table - to substitude multiplication of patch by convolution filter
'''

import torch
from torch.autograd import Function
import  hierarhical_tree_gpu as  ht
import torch.nn.functional as F
import numpy as np


class Quantizer(Function):
    @staticmethod
    def forward(self,  input, ifTraining, tree, lookup_table):
        input_size = input.size()
        if ifTraining:
            # print "Quantizing during training "
            output = quantize(input, tree)
        else:
            # If testing then just precomputed value = conv(center, weights)
            output = self.lookup(input, weights, tree, lookup_table)
        self.save_for_backward(input)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(self, grad_output):
        # quantization module is non-diferantiable, hence, zeroing all gradiens in order not to change the previous layers
        grad_input = grad_output.new_zeros(grad_output.shape)
        return grad_input, None, None, None # as there were 4 inputs, need to provide gradients for all

def quantize_patch(x,y,input, k, kernel):
        stride = 1
        new_stride = 3
        x_step = x*stride
        y_step = y*stride

        patch = input[k, :, x_step:x_step+kernel, y_step:y_step+kernel]
        patch = patch.reshape(1, np.size(patch))
        return patch

def lookup(input, weights, tree, lookup_table):
        pad = 1
        kernel= 5
        stride = 1

        batch_size = input.size(0)
        N = (input.size(3) + 2*pad - kernel)/stride + 1
        output = torch.Tensor(input.size(0), weights.size(0), N, N) # storage for a upsamled input
        input = F.pad(input, (pad,pad,pad,pad))
        for k in range(0, batch_size-1):
            for x in range(0, N):
                for y in range (0,N):
                    x_step = x*stride
                    y_step = y*stride
                    patch = input[k, :, x_step:x_step+kernel, y_step:y_step+kernel]
                    patch = patch.contiguous().view(1, patch.nelement())
                    center, center_id = ht.predict(tree, patch.data.cpu().numpy()) #  quantize currect patch
                    output[k, :, x_step, y_step] = lookup_table[center_id] # copy approximation to upsampled image
        return output

def quantize(input, tree):
        p = 0 # padding, as we want to approximate exact input, we dont add any padding
        kernel= 1
        stride = 1
        batch_size = input.size(0)
        depth = input.size(1)
        m = input.size(3) # width and height of the filter
        if m < kernel:
            kernel = 1

        new_stride = kernel
        N = int((m + 2*p - kernel)/stride + 1)
        output = input.new_zeros((batch_size, depth, kernel*N, kernel*N))
        if p > 0:
            input = F.pad(input, (p,p,p,p))

        for x in range(0, N):
            for y in range (0,N):
                x_step = x*stride
                y_step = y*stride
                xx_step = x*new_stride # step in upsampled image
                yy_stride = y*new_stride

                # if depth == 10:
                patch = input[:, : , x_step:x_step+kernel, y_step:y_step+kernel]
                patch = patch.contiguous().view(batch_size, 1, kernel*kernel*depth)
                center, center_id = ht.predict(tree, patch)
                center = center.view([batch_size, depth, kernel, kernel])

                output[:, :, xx_step:xx_step+kernel, yy_stride:yy_stride+kernel] = center

                
        return output
