import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from region_loss import RegionLoss
from cfg import *
#from layers.batchnorm.bn import BN2d

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks) # merge conv, bn,leaky
        self.loss = self.models[len(self.models)-1]

        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        if self.blocks[(len(self.blocks)-1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

    def forward(self, x):
        ind = -2
        self.loss = None
        outputs = dict()
        for block in self.blocks:
            ind = ind + 1
            #if ind > 0:
            #    return x

            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or block['type'] == 'maxpool' or block['type'] == 'reorg' or block['type'] == 'avgpool' or block['type'] == 'softmax' or block['type'] == 'connected':
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1,x2),1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                print("LOSSS")
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        # print(x.shape)
        return x

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()
    
        prev_filters = 3
        out_filters =[]
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size-1)//2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors)//loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                models.append(loss)
            else:
                print('unknown type %s' % (block['type']))
    
        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))


    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks)-1

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        for blockId in range(1, cutoff+1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    save_fc(fc, model)
                else:
                    save_fc(fc, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()

# def create_modules(module_defs, img_size, cfg):
#     # Constructs module list of layer blocks from module configuration in module_defs

#     img_size = [img_size] * 2 if isinstance(img_size, int) else img_size  # expand if necessary
#     _ = module_defs.pop(0)  # cfg training hyperparams (unused)
#     output_filters = [3]  # input channels
#     module_list = nn.ModuleList()
#     routs = []  # list of layers which rout to deeper layers
#     yolo_index = -1

#     for i, mdef in enumerate(module_defs):
#         modules = nn.Sequential()

#         if mdef['type'] == 'convolutional':
#             bn = mdef['batch_normalize']
#             filters = mdef['filters']
#             k = mdef['size']  # kernel size
#             stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
#             if isinstance(k, int):  # single-size conv
#                 modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
#                                                        out_channels=filters,
#                                                        kernel_size=k,
#                                                        stride=stride,
#                                                        padding=k // 2 if mdef['pad'] else 0,
#                                                        groups=mdef['groups'] if 'groups' in mdef else 1,
#                                                        bias=not bn))
#             else:  # multiple-size conv
#                 modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
#                                                           out_ch=filters,
#                                                           k=k,
#                                                           stride=stride,
#                                                           bias=not bn))

#             if bn:
#                 modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
#             else:
#                 routs.append(i)  # detection output (goes into yolo layer)

#             if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
#                 modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
#             elif mdef['activation'] == 'swish':
#                 modules.add_module('activation', Swish())
#             elif mdef['activation'] == 'mish':
#                 modules.add_module('activation', Mish())

#         elif mdef['type'] == 'BatchNorm2d':
#             filters = output_filters[-1]
#             modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
#             if i == 0 and filters == 3:  # normalize RGB image
#                 # imagenet mean and var https://pytorch.org/docs/stable/torchvision/models.html#classification
#                 modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
#                 modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

#         elif mdef['type'] == 'maxpool':
#             k = mdef['size']  # kernel size
#             stride = mdef['stride']
#             maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
#             if k == 2 and stride == 1:  # yolov3-tiny
#                 modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
#                 modules.add_module('MaxPool2d', maxpool)
#             else:
#                 modules = maxpool

#         elif mdef['type'] == 'upsample':
#             if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
#                 g = (yolo_index + 1) * 2 / 32  # gain
#                 modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))  # img_size = (320, 192)
#             else:
#                 modules = nn.Upsample(scale_factor=mdef['stride'])

#         elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
#             layers = mdef['layers']
#             filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
#             routs.extend([i + l if l < 0 else l for l in layers])
#             modules = FeatureConcat(layers=layers)

#         elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
#             layers = mdef['from']
#             filters = output_filters[-1]
#             routs.extend([i + l if l < 0 else l for l in layers])
#             modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)

#         elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
#             pass

#         elif mdef['type'] == 'yolo':
#             yolo_index += 1
#             stride = [32, 16, 8]  # P5, P4, P3 strides
#             if any(x in cfg for x in ['panet', 'yolov4', 'cd53']):  # stride order reversed
#                 stride = list(reversed(stride))
#             layers = mdef['from'] if 'from' in mdef else []
#             modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
#                                 nc=mdef['classes'],  # number of classes
#                                 img_size=img_size,  # (416, 416)
#                                 yolo_index=yolo_index,  # 0, 1, 2...
#                                 layers=layers,  # output layers
#                                 stride=stride[yolo_index])

#             # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
#             try:
#                 j = layers[yolo_index] if 'from' in mdef else -1
#                 bias_ = module_list[j][0].bias  # shape(255,)
#                 bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
#                 bias[:, 4] += -4.5  # obj
#                 bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
#                 module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
#             except:
#                 print('WARNING: smart bias initialization failure.')

#         else:
#             print('Warning: Unrecognized Layer Type: ' + mdef['type'])

#         # Register module list and number of output filters
#         module_list.append(modules)
#         output_filters.append(filters)

#     routs_binary = [False] * (i + 1)
#     for i in routs:
#         routs_binary[i] = True
#     return module_list, routs_binary

# class YOLOLayer(nn.Module):
#     def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
#         super(YOLOLayer, self).__init__()
#         self.anchors = torch.Tensor(anchors)
#         self.index = yolo_index  # index of this layer in layers
#         self.layers = layers  # model output layer indices
#         self.stride = stride  # layer stride
#         self.nl = len(layers)  # number of output layers (3)
#         self.na = len(anchors)  # number of anchors (3)
#         self.nc = nc  # number of classes (80)
#         self.no = nc + 5  # number of outputs (85)
#         self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
#         self.anchor_vec = self.anchors / self.stride
#         self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

#         if ONNX_EXPORT:
#             self.training = False
#             self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

#     def create_grids(self, ng=(13, 13), device='cpu'):
#         self.nx, self.ny = ng  # x and y grid size
#         self.ng = torch.tensor(ng, dtype=torch.float)

#         # build xy offsets
#         if not self.training:
#             yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
#             self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

#         if self.anchor_vec.device != device:
#             self.anchor_vec = self.anchor_vec.to(device)
#             self.anchor_wh = self.anchor_wh.to(device)

#     def forward(self, p, out):
#         ASFF = False  # https://arxiv.org/abs/1911.09516
#         if ASFF:
#             i, n = self.index, self.nl  # index in layers, number of layers
#             p = out[self.layers[i]]
#             bs, _, ny, nx = p.shape  # bs, 255, 13, 13
#             if (self.nx, self.ny) != (nx, ny):
#                 self.create_grids((nx, ny), p.device)

#             # outputs and weights
#             # w = F.softmax(p[:, -n:], 1)  # normalized weights
#             w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)
#             # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

#             # weighted ASFF sum
#             p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
#             for j in range(n):
#                 if j != i:
#                     p += w[:, j:j + 1] * \
#                          F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)

#         elif ONNX_EXPORT:
#             bs = 1  # batch size
#         else:
#             bs, _, ny, nx = p.shape  # bs, 255, 13, 13
#             if (self.nx, self.ny) != (nx, ny):
#                 self.create_grids((nx, ny), p.device)

#         # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
#         p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

#         if self.training:
#             return p

#         elif ONNX_EXPORT:
#             # Avoid broadcasting for ANE operations
#             m = self.na * self.nx * self.ny
#             ng = 1. / self.ng.repeat(m, 1)
#             grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
#             anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

#             p = p.view(m, self.no)
#             xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
#             wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
#             p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
#                 torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
#             return p_cls, xy * ng, wh

#         else:  # inference
#             io = p.clone()  # inference output
#             io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
#             io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
#             io[..., :4] *= self.stride
#             torch.sigmoid_(io[..., 4:])
#             return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]

# class Darknet(nn.Module):
#     # YOLOv3 object detection model

#     def __init__(self, cfg):
#         super(Darknet, self).__init__()

#         self.module_defs = parse_model_cfg(cfg)
#         self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg)
#         self.yolo_layers = get_yolo_layers(self)
#         # torch_utils.initialize_weights(self)

#         # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
#         self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
#         self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

#     def forward(self, x, augment=False, verbose=False):

#         if not augment:
#             return self.forward_once(x)
#         else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
#             img_size = x.shape[-2:]  # height, width
#             s = [0.83, 0.67]  # scales
#             y = []
#             for i, xi in enumerate((x,
#                                     torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
#                                     torch_utils.scale_img(x, s[1], same_shape=False),  # scale
#                                     )):
#                 # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
#                 y.append(self.forward_once(xi)[0])

#             y[1][..., :4] /= s[0]  # scale
#             y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
#             y[2][..., :4] /= s[1]  # scale

#             # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
#             #     area = yi[..., 2:4].prod(2)[:, :, None]
#             #     if i == 1:
#             #         yi *= (area < 96. ** 2).float()
#             #     elif i == 2:
#             #         yi *= (area > 32. ** 2).float()
#             #     y[i] = yi

#             y = torch.cat(y, 1)
#             return y, None

#     def forward_once(self, x, augment=False, verbose=False):
#         img_size = x.shape[-2:]  # height, width
#         yolo_out, out = [], []
#         if verbose:
#             print('0', x.shape)
#             str = ''

#         # Augment images (inference and test only)
#         if augment:  # https://github.com/ultralytics/yolov3/issues/931
#             nb = x.shape[0]  # batch size
#             s = [0.83, 0.67]  # scales
#             x = torch.cat((x,
#                            torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
#                            torch_utils.scale_img(x, s[1]),  # scale
#                            ), 0)

#         for i, module in enumerate(self.module_list):
#             name = module.__class__.__name__
#             if name in ['WeightedFeatureFusion', 'FeatureConcat']:  # sum, concat
#                 if verbose:
#                     l = [i - 1] + module.layers  # layers
#                     sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
#                     str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
#                 x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
#             elif name == 'YOLOLayer':
#                 yolo_out.append(module(x, out))
#             else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
#                 x = module(x)

#             out.append(x if self.routs[i] else [])
#             if verbose:
#                 print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
#                 str = ''

#         if self.training:  # train
#             return yolo_out
#         elif ONNX_EXPORT:  # export
#             x = [torch.cat(x, 0) for x in zip(*yolo_out)]
#             return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
#         else:  # inference or test
#             x, p = zip(*yolo_out)  # inference output, training output
#             x = torch.cat(x, 1)  # cat yolo outputs
#             if augment:  # de-augment results
#                 x = torch.split(x, nb, dim=0)
#                 x[1][..., :4] /= s[0]  # scale
#                 x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
#                 x[2][..., :4] /= s[1]  # scale
#                 x = torch.cat(x, 1)
#             return x, p

#     def fuse(self):
#         # Fuse Conv2d + BatchNorm2d layers throughout model
#         print('Fusing layers...')
#         fused_list = nn.ModuleList()
#         for a in list(self.children())[0]:
#             if isinstance(a, nn.Sequential):
#                 for i, b in enumerate(a):
#                     if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
#                         # fuse this bn layer with the previous conv2d layer
#                         conv = a[i - 1]
#                         fused = torch_utils.fuse_conv_and_bn(conv, b)
#                         a = nn.Sequential(fused, *list(a.children())[i + 1:])
#                         break
#             fused_list.append(a)
#         self.module_list = fused_list
#         self.info() if not ONNX_EXPORT else None  # yolov3-spp reduced from 225 to 152 layers

#     def info(self, verbose=False):
#         torch_utils.model_info(self, verbose)

# def get_yolo_layers(model):
#     return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]

# def load_darknet_weights(self, weights, cutoff=-1):
#     # Parses and loads the weights stored in 'weights'

#     # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
#     file = Path(weights).name
#     if file == 'darknet53.conv.74':
#         cutoff = 75
#     elif file == 'yolov3-tiny.conv.15':
#         cutoff = 15

#     # Read weights file
#     with open(weights, 'rb') as f:
#         # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
#         self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
#         self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

#         weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

#     ptr = 0
#     for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
#         if mdef['type'] == 'convolutional':
#             conv = module[0]
#             if mdef['batch_normalize']:
#                 # Load BN bias, weights, running mean and running variance
#                 bn = module[1]
#                 nb = bn.bias.numel()  # number of biases
#                 # Bias
#                 bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
#                 ptr += nb
#                 # Weight
#                 bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
#                 ptr += nb
#                 # Running Mean
#                 bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
#                 ptr += nb
#                 # Running Var
#                 bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
#                 ptr += nb
#             else:
#                 # Load conv. bias
#                 nb = conv.bias.numel()
#                 conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
#                 conv.bias.data.copy_(conv_b)
#                 ptr += nb
#             # Load conv. weights
#             nw = conv.weight.numel()  # number of weights
#             conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
#             ptr += nw


# def save_weights(self, path='model.weights', cutoff=-1):
#     # Converts a PyTorch model to Darket format (*.pt to *.weights)
#     # Note: Does not work if model.fuse() is applied
#     with open(path, 'wb') as f:
#         # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
#         self.version.tofile(f)  # (int32) version info: major, minor, revision
#         self.seen.tofile(f)  # (int64) number of images seen during training

#         # Iterate through layers
#         for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
#             if mdef['type'] == 'convolutional':
#                 conv_layer = module[0]
#                 # If batch norm, load bn first
#                 if mdef['batch_normalize']:
#                     bn_layer = module[1]
#                     bn_layer.bias.data.cpu().numpy().tofile(f)
#                     bn_layer.weight.data.cpu().numpy().tofile(f)
#                     bn_layer.running_mean.data.cpu().numpy().tofile(f)
#                     bn_layer.running_var.data.cpu().numpy().tofile(f)
#                 # Load conv bias
#                 else:
#                     conv_layer.bias.data.cpu().numpy().tofile(f)
#                 # Load conv weights
#                 conv_layer.weight.data.cpu().numpy().tofile(f)


# def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
#     # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
#     # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

#     # Initialize model
#     model = Darknet(cfg)

#     # Load weights and save
#     if weights.endswith('.pt'):  # if PyTorch format
#         model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
#         target = weights.rsplit('.', 1)[0] + '.weights'
#         save_weights(model, path=target, cutoff=-1)
#         print("Success: converted '%s' to '%s'" % (weights, target))

#     elif weights.endswith('.weights'):  # darknet format
#         _ = load_darknet_weights(model, weights)

#         chkpt = {'epoch': -1,
#                  'best_fitness': None,
#                  'training_results': None,
#                  'model': model.state_dict(),
#                  'optimizer': None}

#         target = weights.rsplit('.', 1)[0] + '.pt'
#         torch.save(chkpt, target)
#         print("Success: converted '%s' to 's%'" % (weights, target))

#     else:
#         print('Error: extension not supported.')

if __name__ == "__main__":
    model = Darknet("cfg/yolo.cfg").cuda()
    print(model)
    model.load_weights("yolo.weights")
    data = torch.randn(24, 3, 224, 224).cuda()
    output = model(data)
    print(output.size())
