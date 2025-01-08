# Updated TimeMixer with Attention integration
import torch # pytorch的核心库，用于张量操作
import torch.nn as nn # 用于构建神经网络的模块和工具
import torch.nn.functional as F # 提供许多函数用于实现激活函数、损失函数等
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        x = x.transpose(0, 1)  # Convert to (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.transpose(0, 1)  # Back to (batch_size, seq_len, embed_dim)

# 该模块主要用于对时间序列数据进行分解，使用离散傅里叶变换DFT将时间序列数据分解为季节性成分和趋势成分
# 该类继承自nn.Module，表示这是一个神经网络模块
class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    时间序列分解模块
    """

    # top_k = 5 指定要保留的最高频率成分的数量，默认为 5
    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    # 前向传播方法， x 是输入的时间序列数据，通常是一个一维张量
    def forward(self, x):
        # 离散傅里叶变换，计算输入时间序列的快速傅里叶变换（FFT），返回频域表示 xf
        xf = torch.fft.rfft(x)
        # 计算频域表示的幅度，并将直流分量（频率为0的成分）设为0
        freq = abs(xf)
        freq[0] = 0
        # 获取频率幅度中前 top_k 大的成分
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        # 将频率幅度低于前 k 个频率幅度中最小值的成分设为0，以保留较高频的成分
        xf[freq <= top_k_freq.min()] = 0
        # 对修改后的频域表示进行反傅里叶变换，得到季节性成分
        x_season = torch.fft.irfft(xf)
        # 通过从原始时间序列中减去季节性成分来获得趋势成分
        x_trend = x - x_season
        return x_season, x_trend


# 多尺度季节性信息混合
class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    自下而上的混合不同尺度的季节性模式，从高分辨率（细节）逐步传递到低分辨率
    """

    # configs：配置对象，包含了模型的相关超参数（如序列长度、降采样窗口大小等）
    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        """
        降采样层的构建：
        1、每一个模块都包含两个线性层和一个激活函数（用于降采样过程）
        2、每个降采样层逐步降低序列长度，由 configs.down_sampling_window 参数控制降采样窗口的缩放比例
        """
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    # 前向传播：season_list（输入的季节性模式列表，按照从高分辨率到低分辨率的顺序排列）
    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0] # 最高分辨率的季节性模式
        out_low = season_list[1] # 次高分辨率的季节性模式
        out_season_list = [out_high.permute(0, 2, 1)] # 保存每一步的输出
        # [B, L, D] length, dim or channel
        # [B, T, C(sst,sss...), H, W] -> [B, T, C*H*W] 不一定有效果


        """
        多尺度混合：
        1、遍历season_list的各尺度数据，从高到低逐步进行混合
        2、对out_high通过第 i 个降采样层得到 out_low_res，这将 out_high 的信息融合到更低分辨率的 out_low 中
        3、将结果累加到 out_low 中，并更新 out_high
        4、若当前处理的尺度不是最后一个，out_low则更新为下一层的season_list[i + 2]
        5、将每一步的out_high转置后添加到out_season_list中
        """
        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        # 最后返回的是多尺度混合后的季节性模式列表，包含不同尺度的季节性信息，用于后续网络或模型处理
        return out_season_list


# 多尺度趋势信息混合
class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    自上而下混合趋势模式
    从低分辨率（较粗的趋势）向高分辨率（较细的趋势）进行趋势混合
    通过逐层上采样，将低分辨率的趋势模式信息逐步传递到高分辨率中
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        """
        上采样层的构建：
        1、每一个模块包含两个线性层和一个激活函数
        2、每一层将上一层的输入尺寸上采样到更高分辨率
        3、从低分辨率开始构建上采样层，逐步增加分辨率
        """
        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    """
    前向传播：trend_list（输入的趋势模式列表，按照从高分辨率到低分辨率的顺序排列）
    """
    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0] # 初始化为最低分辨率的趋势模式
        out_high = trend_list_reverse[1] # 初始化为次低分辨率的趋势模式
        out_trend_list = [out_low.permute(0, 2, 1)] # 存储每一步的输出

        """
        多尺度混合：
        1、遍历趋势列表中的各尺度数据，从低到高逐步进行融合
        其余操作和混合季节性模式的操作类似
        """
        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        # 因为前面是从低分辨率到高分辨率逐层处理的，因此输出的趋势列表需要再反转，以便符合从高到低的顺序
        out_trend_list.reverse()
        return out_trend_list


# 过去可分解混合模块
class PastDecomposableMixing(nn.Module):
    """
    将分解后的季节性和趋势成分在不同尺度上进行混合，模型可以更好地处理复杂的时间序列数据，尤其是在长时间依赖和多尺度特征方面
    """
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len # 输入的序列长度
        self.pred_len = configs.pred_len # 预测长度
        self.down_sampling_window = configs.down_sampling_window # 降采样的窗口大小，用于在不同尺度上对季节性模式进行处理

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout) # 归一化层和dropout用于规范化和防止过拟合
        self.channel_independence = configs.channel_independence # 通道独立性标志，如果为0，则使用cross_layer对所有通道的数据进行交叉线性变换

        # 时间序列分解方法，根据指定的分解方法，初始化相应的分解模块
        if configs.decomp_method == 'moving_avg': # 移动平均分解，通常用于平滑时间序列数据
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp": # 使用离散傅里叶变换DFT进行分解，提取频域特征
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        # 对季节性和趋势成分进行线性变换
        # 通过一层激活函数和两层线性变换，使得不同通道之间的交叉信息得以传递
        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season，多尺度季节性信息混合
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend，多尺度趋势信息混合
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        # out_cross_layer是最后的输出层，用于整合混合后的季节性和趋势信息
        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    """
    前向传播：x_list（输入的时间序列列表，包含不同的历史时间序列片段）
    1、记录每个时间序列片段的长度
    2、对x_list中的每个时间序列进行分解，得到季节性成分和趋势成分，按需进行交叉线性变换，并保存结果
    3、
    """
    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing，自下向上的多尺度季节性混合
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing，自上而下的多尺度趋势混合
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        # 组合混合后的季节性和趋势成分
        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :]) # 将每个结果的长度截取为原输入的长度，并加入到out_list中

        # 包含不同尺度混合后的时间序列数据
        return out_list


# 该模块主要用于处理不同任务的时间序列数据（包括长期和短期预测、数据补全、异常检测和分类等任务）
# 根据不同任务，模型内部配置了特定的层结构和操作
class Model(nn.Module):
    """
    configs：配置项（其余各层的配置由configs中的参数决定）
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs # 模型配置对象：包含输入序列长度、标签长度、预测长度、是否进行通道独立处理等配置信息
        self.task_name = configs.task_name # 任务名称：用于区分模型的用途，比如时间序列预测、补全、异常检测和分类
        self.seq_len = configs.seq_len # 输入序列长度
        self.label_len = configs.label_len # 标签长度
        self.pred_len = configs.pred_len # 预测长度
        self.down_sampling_window = configs.down_sampling_window # 降采样窗口大小
        self.channel_independence = configs.channel_independence # 是否进行通道独立处理
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs) # 包含多层多尺度分解模块的列表，深度为configs.e_layers
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg) # 时间序列预处理（移动平均分解），用于平滑时间序列数据
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature
        # 根据是否采用通道独立处理来决定使用哪种嵌入方式
        if self.channel_independence == 1:
            # DataEmbedding_wo_pos是一种嵌入层，不包含位置编码，用于将时间序列数据映射到模型的高维表示空间
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers
        # 归一化层: 用于对输入数据进行归一化处理，防止数值过大导致训练不稳定
        # ModuleList包含多个归一化层，数量为 configs.down_sampling_layers + 1，每层配置可以根据 configs.use_norm来决定是否使用归一化
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        """
        预测层：
        1、如果任务是长期或短期预测，模型将构建预测层，用于从模型中提取所需长度的预测序列
        2、predicate_layers：包含多个线性层，对不同分辨率的输入进行预测，长度为 pred_len
        3、projection_layer：投影层，将模型的最后输出通过 projection_layer 映射到预测的目标维度
        4、如果是通道独立处理，则输出通道数为 1， 否则输出维度为 configs.c_out
        5、out_res_layers：用于输出残差层，帮助保持输入和输出在每一层的尺度一致
        6、regression_layers：回归层，用于生成预测的输出序列，适用于回归任务
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
        # 如果任务是数据补全和异常检测，则使用projection_layer将输出映射到所需的维度
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)
        # 如果任务是分类，将使用投影层projection将输出序列压平到d_model * configs.seq_len维度，并映射到类别数num_class上
        # 使用激活函数和dropout层来增强模型的非线性表示能力并防止过拟合
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    """
    该方法对解码器输出dec_out和残差out_res进行处理，并将两者叠加，返回叠加结果
    结合了预测结果和残差信息，使得预测结果能够吸收残差信息，提高输出的准确性
    """
    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out) # 将解码器输出通过投影层投影到目标维度
        out_res = out_res.permute(0, 2, 1) # 将残差进行维度转换，以便与后续层相匹配
        out_res = self.out_res_layers[i](out_res) # 得到进一步处理后的残差信号
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1) # 再将残差通过回归层并调整维度
        dec_out = dec_out + out_res # 将解码器输出和处理后的残差叠加，得到最终的输出
        return dec_out

    """
    该方法对输入数据进行预处理，返回分解后的成分，具体处理取决于channel_independence配置项
    """
    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None) # 表示无需进一步分解
        else:
            out1_list = []
            out2_list = [] # 这两个数组用于存储分解后的成分（趋势成分和季节性成分）
            for x in x_list:
                x_1, x_2 = self.preprocess(x) # 对输入x进行分解，得到趋势和季节性成分
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    """
    该方法用于执行多尺度的下采样处理，根据不同的下采样方法（max、avg或者conv）处理输入x_enc和标记x_mark_enc
    """
    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False) # 最大池化
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window) # 平均池化
        elif self.configs.down_sampling_method == 'conv': # 通过Conv1d卷积层进行下采样，并设置循环填充
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T，调整数据维度，以便执行下采样操作
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = [] # 这两个列表用于保存各尺度的采样结果
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        # 对每一层执行下采样，将每层的采样结果添加到之前创建的两个列表中
        # 主要用于生成不同分辨率的多尺度输入，便于捕捉数据的不同时间尺度特征
        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    """
    负责执行时间序列的预测任务，包含了输入数据的处理、特征转换、编码器（过去可分解混合）、解码器（未来多预测器混合）以及预测结果的生成过程
    该方法的整体结构：
    1、未来时间特征处理：如果启用未来时间特征，先进行特征嵌入
    2、多尺度输入处理：通过下采样操作获得不同尺度的输入数据
    3、标准化和变换：对每个尺度的输入进行标准化，并根据channel_independence配置调整数据维度
    4、嵌入层：通过enc_embedding层将处理后的数据嵌入为模型所需的输入特征
    5、编码器（过去可分解混合模块）：通过pdm_blocks对输入进行处理，提取时间序列的模式（季节性和趋势）
    6、解码器（未来多预测器混合模块）：通过解码器进行未来预测
    7、输出结果处理：将解码器的多个输出进行合并并反标准化，得到最终预测结果
    """
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 判断是否使用未来时间的特征来增强预测
        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        # 对 x_enc, x_mark_enc 进行多尺度下采样处理
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc): # 遍历每个输入的尺度层（x_enc, x_mark_enc）
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm') # 标准化
                if self.channel_independence == 1: # 维度转换，使得数据的通道和时间维度合并，适应模型的需求
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding，将标准化的数据嵌入为模型的输入特征
        enc_out_list = []
        x_list = self.pre_enc(x_list) # 对x_list进行预处理，通常是将时间序列分解为多个部分（如趋势和季节性）
        if x_mark_enc is not None: # 对每个输入序列和对应的时间标记执行嵌入操作，生成编码输出enc_out
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else: # 如果没有时间标记，则仅使用 x 进行嵌入
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past，使用过去可分解混合模块作为编码器，对过去的时间序列数据进行处理
        for i in range(self.layer): # 对每一层进行循环
            enc_out_list = self.pdm_blocks[i](enc_out_list) # 这个处理可能包括序列的分解，季节性和趋势的混合等

        # Future Multipredictor Mixing as decoder for future，使用未来多预测器模块作为解码器，根据过去的信息进行未来的预测
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list) # 根据编码器的输出enc_out_list和输入的历史序列x_list生成多个未来预测结果，B是批量大小

        # 将所有解码器输出叠加（通过求和），并对结果进行反标准化处理
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1) # 将dec_out_list中的所有预测结果沿着最后一个维度堆叠，然后对其进行求和，得到最终的解码器输出
        dec_out = self.normalize_layers[0](dec_out, 'denorm') # 对最终输出执行反标准化denorm，使得输出恢复到原始数据的尺度
        return dec_out

    """
    该方法用来预测未来的时间序列值，生成未来的多步预测
    """
    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list): # 对每个编码器输出进行线性预测并调整维度
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                if self.use_future_temporal_feature: # 如果启用了未来时间特征，则将解码输出dec_out加上未来时间特征嵌入x_mark_dec，并通过projection_layer进一步转换
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)
                # 最后将dec_out重新排列维度以匹配输出格式
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else: # 否则，遍历编码器输出和残差项out_res，在维度对齐后使用out_projection方法进一步转换dec_out并加入dec_out_list
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)
        # 返回的是未来预测的输出列表，其中每个元素代表某个尺度的未来预测
        return dec_out_list

    """
    该方法用于分类任务，将输入数据经过多层处理后输出类别标签
    """
    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None) # 对输入进行多尺度下采样处理
        x_list = x_enc

        # embedding，将输入转换为编码器输入特征
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # 通过pdm_blocks编码层处理，获取最后的编码器输出enc_out
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
        # Output，对输出进行非线性变换和投影
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings，用x_mark_enc作为掩码，将填充位置置零
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)，将输出展平并通过projection线性层映射到类别空间
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        # 返回的是分类的预测输出
        return output

    """
    该方法用于异常检测任务，识别数据中可能存在的异常点
    """
    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None) # 对输入进行多尺度下采样

        x_list = []

        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm') # 数据标准化
            if self.channel_independence == 1: # 通道变换，调整维度
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding，转换为编码器输入特征
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # 通过层级编码，对序列信息进行提取
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # 将编码器的输出映射到指定通道数，重排输出维度并使用denorm进行反标准化
        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        # 包含了检测出的异常信息
        return dec_out

    """
    该方法主要用于处理缺失值补全任务
    首先，通过归一化和标准化对缺失值进行预处理，再将数据编码并输出插值结果
    """
    def imputation(self, x_enc, x_mark_enc, mask):
        # 对每个序列，计算未缺失数据的均值和标准差，用于归一化。计算时，依靠mask判断哪些位置是非缺失的
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        # 使用unsqueeze调整维度
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        # 将 mask 为0的位置替换为0，防止缺失值干扰模型
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev # 对x_enc进行归一化，避免数据分布差异影响模型表现

        # 进行多尺度下采样，获取 x_enc 和 x_mark_enc 的不同尺度表示
        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence == 1: # 根据channel_independence配置将序列转换为特定形状
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1) # 如果有时间特征 x_mark_enc，将其重复以适应数据维度
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]，将x_list中的数据转换为编码表示
            enc_out_list.append(enc_out)

        # 通过 pdm_blocks 进一步提取特征
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # 该层将编码器输出映射为与输入相同的维度
        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        # 最后，将标准差和均值恢复到插值输出上，保证输出符合原始数据分布
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        # 返回值是插值后的数据，可以作为补全后的完整序列
        return dec_out

    """
    该方法是模型的主前向传播方法，根据任务类型选择适当的计算路径。
    负责长短期预测、插值、异常检测或者分类
    """
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError('Other tasks implemented yet')
