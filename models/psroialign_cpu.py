from torch.nn import Conv2d, Linear, Module
from torch.nn import functional as F
from torch.autograd import Function
import torch

class PSROIMaxAlignPooling2D(Function):
    @staticmethod
    def forward(ctx, ps_sensitive_feature, rois, pooling_paras):
        [pooled_channels, pooled_size, spatial_scale, sampling_ratio] = pooling_paras
        channels, height, width = ps_sensitive_feature.shape[1:]
        n_roi = rois.shape[0]
        pooled_data = torch.zeros([n_roi, pooled_channels, pooled_size, pooled_size], dtype=torch.float)
        '''
        ps_sensitive_feature : [1, 2048, hh, ww] torch.tensor
        rois : [R,4] np.array
        '''
        w1w2w3w4_group = []
        xlowylowxhighyhigh_group = []
        for i in range(pooled_data.nelement()):
                
            pw = i % pooled_size
            ph = (i // pooled_size) % pooled_size
            ctop = (i // pooled_size // pooled_size) % pooled_channels
            n = (i // pooled_size // pooled_size // pooled_channels)
            roi_start_h = rois[n, 0] * spatial_scale
            roi_start_w = rois[n, 1] * spatial_scale
            roi_end_h = rois[n, 2] * spatial_scale
            roi_end_w = rois[n, 3] * spatial_scale

            roi_height = max(roi_end_h - roi_start_h, 1.)
            roi_width = max(roi_end_w - roi_start_w, 1.)
            bin_size_h = 1. * roi_height / pooled_size
            bin_size_w = 1. * roi_width / pooled_size
            
            c = (ctop * pooled_size + ph) * pooled_size + pw
            
            if sampling_ratio > 0:
                roi_bin_grid_h = sampling_ratio
                roi_bin_grid_w = sampling_ratio
            else:
                roi_bin_grid_h = np.ceil(roi_height / pooled_size)
                roi_bin_grid_w = np.ceil(roi_width / pooled_size)

            maxval = -1e20
            w1w2w3w4 = None
            xlowylowxhighyhigh = None
            
            iy = 0            
            while iy < roi_bin_grid_h:
                y = roi_start_h + ph * bin_size_h + (iy + .5) * bin_size_h / roi_bin_grid_h
                ix = 0
                while ix < roi_bin_grid_w:
                    x = roi_start_w + pw * bin_size_w + (ix + .5) * bin_size_w / roi_bin_grid_w

                    # bilinear interpolation {{
#                     if y < -1 or y > height or x < -1 or x > width:
#                         # empty
#                         continue
#                         print('not in range, x:{}, y:{}'.format(x,y))

                    if y <= 0:
                        y = 0
                    if x <= 0:
                        x = 0

                    y_low = int(y)
                    x_low = int(x)

                    if y_low >= height - 1:
                        y_high = y_low = height - 1
                        y = float(y_low)
                    else:
                        y_high = y_low + 1

                    if x_low >= width - 1:
                        x_high = x_low = width - 1
                        x = float(x_low)
                    else:
                        x_high = x_low + 1

                    ly = y - y_low
                    lx = x - x_low
                    hy = 1. - ly
                    hx = 1. - lx
                    
                    v1 = ps_sensitive_feature[0, c, y_low, x_low].item()
                    v2 = ps_sensitive_feature[0, c, y_low, x_high].item()
                    v3 = ps_sensitive_feature[0, c, y_high, x_low].item()
                    v4 = ps_sensitive_feature[0, c, y_high, x_high].item()
                    
                    w1 = hy * hx
                    w2 = hy * lx
                    w3 = ly * hx
                    w4 = ly * lx
                    
                    tmpval = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
                    if (tmpval > maxval):
                        maxval = tmpval
                        max_w1w2w3w4 = torch.tensor([[float(w1), float(w2), float(w3), float(w4)]])
                        max_xlowylowxhighyhigh = torch.tensor([[x_low, y_low, x_high, y_high]], dtype=torch.long)
                        
                    ix += 1
                iy += 1
            
            print(max_xlowylowxhighyhigh)
            xlowylowxhighyhigh_group.append(max_xlowylowxhighyhigh)
            w1w2w3w4_group.append(max_w1w2w3w4)
            print(max_w1w2w3w4)  
            pooled_data[n, ctop, ph, pw] = maxval  
            
        ctx.save_for_backward(torch.cat(xlowylowxhighyhigh_group), torch.cat(w1w2w3w4_group), torch.tensor([channels, height, width]))
        return pooled_data
    
    @staticmethod
    def backward(ctx, grad_output):
        xlowylowxhighyhigh_group, w1w2w3w4_group, paras = ctx.saved_tensors
        channels, height, width = paras[0].item(), paras[1].item(), paras[2].item()
        pooled_channels, pooled_size = grad_output.shape[1], grad_output.shape[2]
        bp_grads = torch.zeros([1, channels, height, width], dtype=torch.float)
        
        for i in range(grad_output.nelement()):
            
            pw = i % pooled_size
            ph = (i // pooled_size) % pooled_size
            ctop = (i // pooled_size // pooled_size) % pooled_channels
            n = (i // pooled_size // pooled_size // pooled_channels)
            
            c = (ctop * pooled_size + ph) * pooled_size + pw
            
            [w1, w2, w3, w4] = w1w2w3w4_group[i].tolist()
            print([w1, w2, w3, w4])
            [x_low, y_low, x_high, y_high] = xlowylowxhighyhigh_group[i].tolist()
            print([x_low, y_low, x_high, y_high])
            grad_this_bin = grad_output[n, ctop, ph, pw]
        
            g1 = grad_this_bin * w1
            g2 = grad_this_bin * w2
            g3 = grad_this_bin * w3
            g4 = grad_this_bin * w4
            
            if (x_low >= 0 and x_high >= 0 and y_low >= 0 and y_high >= 0):
                bp_grads[0, c, y_low, x_low] += g1
                bp_grads[0, c, y_low, x_high] += g2
                bp_grads[0, c, y_high, x_low] += g3
                bp_grads[0, c, y_high, x_high] += g4
            else:
                print('backward bypass negative coordinates')
        return bp_grads, None, None