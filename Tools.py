import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#####################
#### CV  Modules ####
#####################
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
        self.conv2 = nn.Conv2d(outChannels, outChannels,kernel_size=3, stride = stride, groups = outChannels, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannels ,eps=1e-3, momentum= 1- 0.99)
        self.conv3 = nn.Conv2d(outChannels, outChannels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(outChannels)
        self.relu1 = MemoryEfficientSwish()
        self.relu2 = MemoryEfficientSwish()
        self.downSample = nn.Sequential(nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1),
                                        nn.BatchNorm2d(outChannels,eps=1e-3, momentum= 1- 0.99),
                                        MemoryEfficientSwish())

    def forward(self, x):
        identity = self.downSample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity

        return out


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
        self.liner = nn.Sequential(nn.Linear(embed_size,hidden_size),
                                   nn.BatchNorm1d(hidden_size),
                                   Mish())

    def forward(self, x):
        """
        :param x: [batch,sequences]
        :return: [batch,sequences,hidden_size]
        """
        embedTensor = self.embeddingLayer(x)
        linerTensor = self.liner(embedTensor)
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
##### Other funtions #####
##########################

def AddN(tensorList : []):
    if len(tensorList)==1:
        return tensorList[0]
    else:
        addR = tensorList[0] + tensorList[1]
        for i in range(2,len(tensorList)):
            addR = addR + tensorList[i]
        return addR