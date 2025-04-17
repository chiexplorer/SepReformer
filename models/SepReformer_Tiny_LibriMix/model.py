import sys
sys.path.append('../')

import torch
import warnings
warnings.filterwarnings('ignore')

from utils.decorators import *
from .modules.module import *  # 全局运行
# from modules.module import *   # 当前文件调试

@logger_wraps()
class Model(torch.nn.Module):
    def __init__(self, 
                 num_stages: int, 
                 num_spks: int, 
                 module_audio_enc: dict, 
                 module_feature_projector: dict, 
                 module_separator: dict, 
                 module_output_layer: dict, 
                 module_audio_dec: dict):
        super().__init__()
        self.num_stages = num_stages
        self.num_spks = num_spks
        self.audio_encoder = AudioEncoder(**module_audio_enc)
        self.feature_projector = FeatureProjector(**module_feature_projector)
        self.separator = Separator(**module_separator)
        self.out_layer = OutputLayer(**module_output_layer)
        self.audio_decoder = AudioDecoder(**module_audio_dec)
        
        # Aux_loss
        self.out_layer_bn = torch.nn.ModuleList([])
        self.decoder_bn = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.out_layer_bn.append(OutputLayer(**module_output_layer, masking=True))
            self.decoder_bn.append(AudioDecoder(**module_audio_dec))
        
    def forward(self, x):
        encoder_output = self.audio_encoder(x)
        projected_feature = self.feature_projector(encoder_output)
        last_stage_output, each_stage_outputs = self.separator(projected_feature)
        out_layer_output = self.out_layer(last_stage_output, encoder_output)
        each_spk_output = [out_layer_output[idx] for idx in range(self.num_spks)]
        audio = [self.audio_decoder(each_spk_output[idx]) for idx in range(self.num_spks)]
        
        # Aux_loss
        audio_aux = []
        for idx, each_stage_output in enumerate(each_stage_outputs):
            each_stage_output = self.out_layer_bn[idx](torch.nn.functional.upsample(each_stage_output, encoder_output.shape[-1]), encoder_output)
            out_aux = [each_stage_output[jdx] for jdx in range(self.num_spks)]
            audio_aux.append([self.decoder_bn[idx](out_aux[jdx])[...,:x.shape[-1]] for jdx in range(self.num_spks)])
            
        return audio, audio_aux


if __name__ == '__main__':
    from thop import profile
    from torchinfo import summary
    from ptflops import get_model_complexity_info

    num_stages = 4
    num_spks = 2
    module_audio_enc = {'bias': False, 'groups': 1, 'in_channels': 1, 'kernel_size': 16, 'out_channels': 256, 'stride': 4}
    module_feature_projector = {'bias': False, 'in_channels': 256, 'kernel_size': 1, 'num_channels': 256, 'out_channels': 64}
    module_separator = {'num_stages': 4, 'relative_positional_encoding': {'in_channels': 64, 'num_heads': 8, 'maxlen': 2000, 'embed_v': False}, 'enc_stage': {'global_blocks': {'in_channels': 64, 'num_mha_heads': 8, 'dropout_rate': 0.05}, 'local_blocks': {'in_channels': 64, 'kernel_size': 65, 'dropout_rate': 0.05}, 'down_conv_layer': {'in_channels': 64, 'samp_kernel_size': 5}}, 'spk_split_stage': {'in_channels': 64, 'num_spks': 2}, 'simple_fusion': {'out_channels': 64}, 'dec_stage': {'num_spks': 2, 'global_blocks': {'in_channels': 64, 'num_mha_heads': 8, 'dropout_rate': 0.05}, 'local_blocks': {'in_channels': 64, 'kernel_size': 65, 'dropout_rate': 0.05}, 'spk_attention': {'in_channels': 64, 'num_mha_heads': 8, 'dropout_rate': 0.05}}}
    module_output_layer = {'in_channels': 256, 'num_spks': 2, 'out_channels': 64}
    module_audio_dec = {'bias': False, 'in_channels': 256, 'kernel_size': 16, 'out_channels': 1, 'stride': 4}
    model = Model(num_stages, num_spks, module_audio_enc, module_feature_projector, module_separator, module_output_layer, module_audio_dec)

    audio_len = 16000
    x = torch.randn((2, audio_len))
    # y = model(x)
    # print(y.shape)

    # 模型复杂度
    macs, params = profile(model, inputs=(x, ))
    mb = 1000*1000
    print(f"MACs: [{macs/mb/1000}] G \nParams: [{params/mb}] M")
    # 计算参数量
    print("模型参数量详情：")
    summary(model, input_size=(1, audio_len), mode="train")