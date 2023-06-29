import torch
import torch.nn.functional as F

class IMV_F(torch.nn.Module):
    def __init__(self):
        super(IMV_F, self).__init__()
        self.epsilon = 1e-5

    def Inf_Vis_Salient_Background(self, img_ir, img_vi, img_ir_mask):
        img_ir_ahead = torch.mul(img_ir_mask, img_ir)
        img_ir_back = torch.mul(1 - img_ir_mask, img_ir)

        img_vi_ahead = torch.mul(img_ir_mask, img_vi)
        img_vi_back = torch.mul(1 - img_ir_mask, img_vi)

        return img_ir_ahead, img_ir_back, img_vi_ahead, img_vi_back

    def Salient_Background_Fusion(self, en_ir_a, en_ir_b, en_vi_a, en_vi_b): #Salient:en_ir_a and en_vi_a.
        SCA = Spatial_Channel_Attention()
        fusion_function = SCA.attention_fusion_weight

        SCA_1 = fusion_function(en_ir_b[0], en_vi_b[0])
        SCA_2 = fusion_function(en_ir_b[1], en_vi_b[1])
        SCA_3 = fusion_function(en_ir_b[2], en_vi_b[2])
        SCA_4 = fusion_function(en_ir_b[3], en_vi_b[3])

        SCA_weight_1 = SCA_1 / (SCA_1 + en_vi_b[0] + 1e-5)
        SCA_weight_2 = SCA_2 / (SCA_2 + en_vi_b[1] + 1e-5)
        SCA_weight_3 = SCA_3 / (SCA_3 + en_vi_b[2] + 1e-5)
        SCA_weight_4 = SCA_4 / (SCA_4 + en_vi_b[3] + 1e-5)

        # The parameter μ=0.8, the optimal value obtained by the ablation experiment of Eq.(9) in the paper.
        s_miu = 0.8

        # fx_0 = (addition of significance) + 0.6 * (softmax for vi_b and SCA) + 0.4 * vi_b
        f1_0 = (s_miu * en_ir_a[0] + (1 - s_miu) * en_vi_a[0]
                ) + 0.6 * (SCA_weight_1 * SCA_1 + (1 - SCA_weight_1) * en_vi_b[0]) + 0.4 * en_vi_b[0]
        f2_0 = (s_miu * en_ir_a[1] + (1 - s_miu) * en_vi_a[1]
                ) + 0.6 * (SCA_weight_2 * SCA_2 + (1 - SCA_weight_2) * en_vi_b[1]) + 0.4 * en_vi_b[1]
        f3_0 = (s_miu * en_ir_a[2] + (1 - s_miu) * en_vi_a[2]
                ) + 0.6 * (SCA_weight_3 * SCA_3 + (1 - SCA_weight_3) * en_vi_b[2]) + 0.4 * en_vi_b[2]
        f4_0 = (s_miu * en_ir_a[3] + (1 - s_miu) * en_vi_a[3]
                ) + 0.6 * (SCA_weight_4 * SCA_4 + (1 - SCA_weight_4) * en_vi_b[3]) + 0.4 * en_vi_b[3]

        return f1_0, f2_0, f3_0, f4_0


# SCA (Spatial channel attention) module
class Spatial_Channel_Attention(torch.nn.Module):
    def __init__(self):
        super(Spatial_Channel_Attention, self).__init__()
        self.epsilon = 1e-5
        self.global_pooling_type = 'average_global_pooling' # channel_attention
        self.spatial_type = 'mean' # spatial_attention

    # attention fusion strategy
    def attention_fusion_weight(self, tensor1, tensor2):
        f_channel = self.channel_fusion(tensor1, tensor2)
        f_spatial = self.spatial_fusion(tensor1, tensor2)

        tensor_fusion = (f_channel * 3 + f_spatial) / 4

        return tensor_fusion

    def channel_fusion(self, tensor1, tensor2):
        shape = tensor1.size()
        # calculate channel attention
        global_p1 = self.channel_attention(tensor1)
        global_p2 = self.channel_attention(tensor2)

        # get weight map
        global_p_w1 = global_p1 / (global_p1 + global_p2 + self.epsilon)
        global_p_w2 = global_p2 / (global_p1 + global_p2 + self.epsilon)

        global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
        global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

        tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

        return tensor_f

    def spatial_fusion(self, tensor1, tensor2):
        shape = tensor1.size()
        # calculate spatial attention
        spatial1 = self.spatial_attention(tensor1)
        spatial2 = self.spatial_attention(tensor2)

        # get weight map, soft-max
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + self.epsilon)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + self.epsilon)

        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

        tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

        return tensor_f

    def channel_attention(self, tensor):
        # global pooling
        shape = tensor.size()

        if self.global_pooling_type is 'average_global_pooling':
            pooling_function = F.avg_pool2d
        elif self.global_pooling_type is 'max_global_pooling':
            pooling_function = F.max_pool2d
        global_p = pooling_function(tensor, kernel_size=shape[2:])
        return global_p

    def spatial_attention(self, tensor):
        spatial = []
        if self.spatial_type is 'mean':
            spatial = tensor.mean(dim=1, keepdim=True)
        elif self.spatial_type is 'sum':
            spatial = tensor.sum(dim=1, keepdim=True)
        return spatial


if __name__ == '__main__':
    sca = Spatial_Channel_Attention().attention_fusion_weight
    a = torch.randn(4, 1, 256, 256)
    b = torch.randn(4, 1, 256, 256)
    SCA_1 = sca(a, b)
    print(SCA_1.shape)

