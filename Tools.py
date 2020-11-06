import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#####################
#### CV  Modules ####
#####################

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, stride=stride, padding=1)
        self.p_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)
        output = self.conv(x_offset)

        return output

    def _get_p_n(self, N, dtype, device):
        p_n_x, p_n_y = torch.meshgrid(
            [torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1)])
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x.to(device)), torch.flatten(p_n_y.to(device))], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype, device):
        p_0_x, p_0_y = torch.meshgrid(
            [torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride)])
        p_0_x = torch.flatten(p_0_x.to(device)).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y.to(device)).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        device = offset.device
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype, device)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype, device)
        # print(p_0.device)
        # print(p_n.device)
        # print(offset.device)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset


class Pool2dStaticSamePadding(nn.Module):
    """
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, kernel_size, stride ,pooling = "avg"):
        super().__init__()
        if pooling.lower() == "max":
            self.pool = nn.MaxPool2d(kernel_size=kernel_size,stride=stride)
        elif pooling.lower() == "avg":
            self.pool = nn.AvgPool2d(kernel_size=kernel_size,stride=stride,ceil_mode=True, count_include_pad=False)
        else:
            raise Exception("No implement.")
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        h_step = math.ceil(torch.true_divide(w , self.stride[1]))
        v_step = math.ceil(torch.true_divide(h , self.stride[0]))
        h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
        v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)

        extra_h = h_cover_len - w
        extra_v = v_cover_len - h

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x



class Conv2dDynamicSamePadding(nn.Module):
    """
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        h_step = math.ceil(torch.true_divide(w ,self.stride[1]))
        v_step = math.ceil(torch.true_divide(h , self.stride[0]))
        h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
        v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)
        extra_h = h_cover_len - w
        extra_v = v_cover_len - h
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top
        x = F.pad(x, [left, right, top, bottom])
        x = self.conv(x)
        return x


from torch.nn import Parameter
class DynamicConv2d(nn.Module):

    def __init__(self,in_channels, out_channels,kernel_size, groups, stride, padding, tau = 1., k = 4):
        super().__init__()
        self.weight = Parameter(torch.ones(size=[k, out_channels, in_channels // groups, kernel_size, kernel_size],
                                           dtype=torch.float32,requires_grad=True),requires_grad=True)
        self.bias = Parameter(torch.zeros(size=[k, out_channels],requires_grad=True),requires_grad=True)
        self.stride = [stride,stride]
        self.padding = [padding,padding]
        self.dense1 = nn.Conv2d(in_channels, in_channels // 2,kernel_size=1,stride=1,padding=0)
        self.dense2 = nn.Conv2d(in_channels // 2, k ,kernel_size=1,stride=1,padding=0)
        self.tau = tau
        self.k = k
        self.groups = groups

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self,x):
        globalAvg = F.adaptive_avg_pool2d(x,[1,1])
        dense1 = F.relu(self.dense1(globalAvg),True)
        dense2 = self.dense2(dense1)
        softmaxT = torch.softmax(torch.div(dense2,self.tau),dim=-3)
        batch = x.size(0)
        batchOut = []
        for i in range(batch):
            oneSample = x[i].unsqueeze(dim=0)
            softmaxTi = softmaxT[i]
            #print(softmaxTi)
            attentionW = softmaxTi.view(size=[self.k,1,1,1,1])
            attWeight = (self.weight * attentionW).sum(dim=0,keepdim=False)
            attentionB = softmaxTi.view(size=[self.k,1])
            attBias = (self.bias * attentionB).sum(dim=0,keepdim=False)
            batchOut.append(F.conv2d(oneSample, attWeight, attBias, self.stride,
                        self.padding, groups= self.groups))
        return torch.cat(batchOut,dim=0)

class SeparableConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1, norm= False, activation=False):
        super(SeparableConvBlock, self).__init__()
        self._bn_mom = 1 - 0.99 # pytorch's difference from tensorflow
        self._bn_eps = 1e-3

        self.depthwise_conv = Conv2dDynamicSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=stride, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dDynamicSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(out_channels, momentum=self._bn_mom, eps= self._bn_eps)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)
        return x

class Bottleneck(nn.Module):

    def __init__(self, inChannels, outChannels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(outChannels, eps=1e-3, momentum= 1- 0.99)
        self.conv2 = Conv2dDynamicSamePadding(outChannels, outChannels,kernel_size=3, stride = stride, groups = outChannels)
        self.bn2 = nn.BatchNorm2d(outChannels ,eps=1e-3, momentum= 1- 0.99)
        self.conv3 = nn.Conv2d(outChannels, outChannels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(outChannels)
        self.relu1 = MemoryEfficientSwish()
        self.relu2 = MemoryEfficientSwish()
        self.downSample = nn.Sequential(Conv2dDynamicSamePadding(inChannels, outChannels, kernel_size=3, stride=stride),
                                        nn.BatchNorm2d(outChannels))

    def forward(self, x):
        identity = self.downSample(x)

        outb = self.conv1(x)
        outb = self.bn1(outb)
        outb = self.relu1(outb)

        outb = self.conv2(outb)
        outb = self.bn2(outb)
        outb = self.relu2(outb)

        outb = self.conv3(outb)
        outb = self.bn3(outb)

        outb += identity

        return outb


######################################
######### Activation Modules #########
######################################
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):

    def __init__(self):
        super(Mish,self).__init__()

    def forward(self,x):
        return x * torch.tanh(F.softplus(x))


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SwishImplementation.apply(x)



#############################
##### Attention Modules #####
#############################
class SE(nn.Module):

    def __init__(self,in_channels,out_channels,reduce_factor = 2):
        super().__init__()
        self.dense1 = nn.Conv2d(in_channels,out_channels=in_channels // reduce_factor,kernel_size=1,
                                stride=1,padding=0)
        self.dense2 = nn.Conv2d(in_channels // reduce_factor,out_channels=out_channels,kernel_size=1,
                                stride=1,padding=0)
        self.act = MemoryEfficientSwish()

    def forward(self,x):
        globalPooling = F.adaptive_avg_pool2d(x,[1,1])
        dense1 = self.act((self.dense1(globalPooling)))
        dense2 = self.dense2(dense1)
        return torch.sigmoid(dense2)

class ChannelsAttention(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    """
    def __init__(self, in_channels, out_channels, reduce_factor = 2):
        super(ChannelsAttention, self).__init__()
        self.dense1 = nn.Linear(in_channels, in_channels // reduce_factor)
        self.dense2 = nn.Linear(in_channels // reduce_factor, out_channels)
        self.inc = in_channels
        self.red = reduce_factor
        self.flatten = nn.Flatten()

    def forward(self, inp):
        maxPool = self.flatten(F.adaptive_max_pool2d(inp, output_size=[1,1]))
        avgPool = self.flatten(F.adaptive_avg_pool2d(inp, output_size=[1,1]))
        maxDense = self.dense2(F.relu(self.dense1(maxPool)))
        avgDense = self.dense2(F.relu(self.dense1(avgPool)))
        denseAdd = maxDense + avgDense
        _, c = denseAdd.shape
        denseAdd = torch.reshape(denseAdd, shape=[-1,c,1,1])
        return torch.sigmoid(denseAdd)

class SpatialAttention(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    """

    def __init__(self, in_channels, reduce_factor = 5):
        super(SpatialAttention, self).__init__()
        self.conv1 = Conv2dDynamicSamePadding(in_channels // reduce_factor * 2, in_channels // reduce_factor,
                                              kernel_size=3)
        self.conv2 = Conv2dDynamicSamePadding(in_channels // reduce_factor, 1,
                                              kernel_size=3)
        self.in_channels = in_channels
        self.reduce_factor = reduce_factor
        self.maxPool = nn.AdaptiveMaxPool1d(output_size=[in_channels // reduce_factor])
        self.avgPool = nn.AdaptiveAvgPool1d(output_size=[in_channels // reduce_factor])

    def forward(self, inp):
        _, c, h, w = inp.shape
        oneDTensor = torch.reshape(inp, shape=[-1, h * w, c])
        maxPool = self.maxPool(oneDTensor)
        avgPool = self.avgPool(oneDTensor)
        catTensor = torch.cat([maxPool, avgPool], dim=-1).view([-1, self.in_channels // self.reduce_factor * 2, h, w])
        return torch.sigmoid(self.conv2(F.relu(self.conv1(catTensor))))


class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None , mode='embedded', bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        :param in_channels: original channel size (1024 in the paper)
        :param inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
        :param mode: `gaussian`, `embedded`, `dot`
        :param bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        if mode not in ['gaussian', 'embedded', 'dot']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot`')

        self.mode = mode

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, batch norm layers for different dimensions
        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)


    def forward(self, x):
        """
        args
            x: (N, C, H, W) for dimension 2
        """

        batch_size = x.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        else:
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        else:
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z

#######################
##### NLP Modules #####
#######################
class FactorizedEmbedding(nn.Module):

    def __init__(self,vocab_size,embed_size,hidden_size):
        """
        :param vocab_size:
        :param embed_size:
        :param hidden_size: hidden_size must much larger than embed_size
        """
        super(FactorizedEmbedding,self).__init__()
        self.embeddingLayer = nn.Embedding(vocab_size,embed_size)
        self.linear = nn.Sequential(nn.Linear(embed_size,hidden_size),
                                    nn.BatchNorm1d(hidden_size))

    def forward(self, x):
        embedTensor = self.embeddingLayer(x).squeeze(dim=1)
        linerTensor = self.linear(embedTensor)
        return linerTensor

######################################
##### Prevent overfitted Modules #####
######################################

def drop_connect(inputs, p, training):
    """Drop connect.
    Args:
        inputs (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.
    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output

##########################
##### Other functions #####
##########################

def AddN(tensorList : []):
    if len(tensorList)==1:
        return tensorList[0]
    else:
        addR = tensorList[0] + tensorList[1]
        for i in range(2,len(tensorList)):
            addR = addR + tensorList[i]
        return addR

def update_bn(loader, model):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Arguments:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for inputs in loader:
        model(inputs[0], inputs[1])

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

###################
### TEST MODULE ###
###################
def hook_fn(module, g_input, g_output):
    ### g_input contains the partial derivative for weight W and bias B.
    ### g_output contains the partial derivative for input x.
    print("Module Name {}".format(module.name))
    print("Grad input is {}".format(g_input))
    print("Grad output is {}".format(g_output))
    g_input = list((g_input[i] for i in range(len(g_input))))
    g_output = list((g_output[i] * 0.1 for i in range(len(g_output))))
    return g_output[0], g_input[1]
class Linear(nn.Module):
    def __init__(self,in_dim, out_dim, name):
        super(Linear, self).__init__()
        self.name = name
        self.register_parameter("weight", nn.Parameter(data=torch.ones([in_dim, out_dim]),requires_grad=True))
        self.register_parameter("bias", nn.Parameter(data=torch.zeros([out_dim]), requires_grad=True))
    def forward(self,x):
        print(self.name + " input x requires grad {}".format(x.requires_grad))
        return torch.add(torch.matmul(x,self.weight),self.bias)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = Linear(1,1,"linear1")
        self.linear2 = Linear(1,1,"linear2")
        self.linear1.register_backward_hook(hook_fn)
        self.linear2.register_backward_hook(hook_fn)
    def forward(self,x):
        return self.linear2(self.linear1(x))
if __name__ == "__main__":
    testInput = torch.ones([1, 1]).float()
    testModule = Net()
    loss = nn.MSELoss(reduction="sum")
    out = testModule(testInput)
    lossVal = loss(out, torch.zeros([1, 1]))
    print("Loss value is {}".format(lossVal.tolist()))
    print("Net output is {}".format(out.tolist()))
    lossVal.backward()