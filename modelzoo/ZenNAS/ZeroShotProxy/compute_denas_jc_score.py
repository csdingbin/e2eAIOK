'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from
https://github.com/SamsungLabs/zero-cost-nas
'''


# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np



import torch

def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net


def get_batch_jacobian(net, x):
    # inputs = inputs.double()
    split_data = 1
    x.requires_grad_(True)

    N = x.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data
        y = net(x[st:en].double())
        y.backward(torch.ones_like(y))

    jacob = x.grad.detach()
    x.requires_grad_(False)
    return jacob

def eval_score(jacob):
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))


def compute_jacob_cov(net, inputs):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()

    jacobs = get_batch_jacobian(net, inputs)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    try:
        jc = eval_score(jacobs)
    except Exception as e:
        print(e)
        jc = np.nan

    return jc


def get_model_latency(model, inputs, repeat_times):
    
    device = torch.device('cpu')
    dtype = torch.float32
    inputs = inputs.double()
    model.eval()
    warmup_T = 1
    with torch.no_grad():
        for i in range(warmup_T):
            the_output = model(inputs)
        start_timer = time.time()
        for repeat_count in range(repeat_times):
            the_output = model(inputs)
    end_timer = time.time()
    the_latency = (end_timer - start_timer) / float(repeat_times)
    return the_latency

def get_layer_metric_array(net, metric, mode):
    metric_array = []

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))

    return metric_array







def compute_synflow_per_weight(net, inputs, mode):
    device = inputs.device

    # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net)

    # Compute gradients with input of 1s
    net.zero_grad()
    net.double()
    input_dim = list(inputs[0, :].shape)
    inputs = torch.ones([1] + input_dim).double().to(device)
    output = net.forward(inputs)
    torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, synflow, mode)

    # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs





def do_compute_nas_score(gpu, model, resolution, batch_size, mixup_gamma, belta):
    dtype = torch.float32
    network_weight_gaussian_init(model)
    input = torch.randn(size=[batch_size, 3, resolution, resolution],  dtype=dtype)
    input2 = torch.randn(size=[batch_size, 3, resolution, resolution], dtype=dtype)
    mixup_input = input + mixup_gamma * input2

    output = model.forward_pre_GAP(input)
    mixup_output = model.forward_pre_GAP(mixup_input)

    nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
    nas_score = torch.mean(nas_score)

    model.train()
    model.requires_grad_(True)

    model.zero_grad()

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    

    if gpu is not None:
        input = input.cuda(gpu)

    grads_abs_list = compute_synflow_per_weight(net=model, inputs=input, mode='')
   
    score = 0
    for grad_abs in grads_abs_list:
        if len(grad_abs.shape) == 4:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1,2,3])))
        elif len(grad_abs.shape) == 2:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
        else:
            raise RuntimeError('!!!')

    jc = compute_jacob_cov(model, input)
    latency = get_model_latency(model=model, inputs=input, repeat_times=1) 
    

    nas_score = torch.log(nas_score) + ( - 1 * score) - float(jc)
    nas_score = nas_score/(1+latency*100*belta) 


    return nas_score

