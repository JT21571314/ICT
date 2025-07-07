import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from torch.nn import CrossEntropyLoss
import torchvision
import numpy as np
import os
from transformers import RobertaModel,BertModel,AlbertModel,ElectraModel,ViTModel,SwinModel,DeiTModel,ConvNextModel
from .new_model4 import InterlanceDecoder
from timm.models.vision_transformer import Mlp
class ICTModel(nn.Module):
    def __init__(self,config1,config2,text_num_labels,alpha,beta,text_model_name="roberta",image_model_name='vit'):
        super().__init__()
        if text_model_name == 'roberta' or text_model_name == 'roberta-large':
            self.roberta = RobertaModel(config1,add_pooling_layer=False)
        elif text_model_name == 'bert':
            self.bert = BertModel(config1, add_pooling_layer=False)
        elif text_model_name == 'albert':
            self.albert = AlbertModel(config1, add_pooling_layer=False)
        elif text_model_name == 'electra':
            self.electra = ElectraModel(config1)
        if image_model_name == 'vit' or image_model_name == "vit-large":
            self.vit = ViTModel(config2)
        elif image_model_name == 'swin':
            self.swin = SwinModel(config2)
        elif image_model_name == 'deit':
            self.deit = DeiTModel(config2)
        elif image_model_name == 'convnext':
            self.convnext = ConvNextModel(config2)
        self.alpha = alpha
        self.beta = beta
        self.text_model_name=text_model_name
        self.image_model_name=image_model_name
        self.config1 = config1
        self.config2 = config2
        self.gelu0 = nn.GELU()
        self.gelu1 = nn.GELU()
        self.gelu = nn.GELU()
        self.ys_t = nn.Linear(60, 128)
        self.ys_i = nn.Linear(197, 128)
        self.text_num_labels = text_num_labels
        self.image_text_cross = InterlanceDecoder(embed_dim=config1.hidden_size,num_classes=self.text_num_labels,num_heads=16,mlp_ratio=4.0, drop_rate=0.1,attn_drop_rate=0.1,drop_path_rate=0.0)
        self.dropout = nn.Dropout(config1.hidden_dropout_prob)
        self.loss_fct = CrossEntropyLoss()
        self.mlp0 = Mlp(in_features=config1.hidden_size, hidden_features=int(config1.hidden_size*2))
        
        self.classifier1 = nn.Linear(config1.hidden_size, self.text_num_labels)
        self.classifier0= Mlp(in_features=config1.hidden_size, out_features=self.text_num_labels, hidden_features=int(config1.hidden_size*2))

        self.CRF = CRF(self.text_num_labels, batch_first=True)
        self.loss_dst = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.9]), reduction='mean') 

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                pixel_values=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                image_labels=None,
                head_mask=None,
                cross_labels=None,
                return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config1.use_return_dict
        if self.text_model_name == 'bert':
            text_outputs = self.bert(input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
        elif self.text_model_name == 'roberta' or self.text_model_name == 'roberta-large':
            text_outputs = self.roberta(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        elif self.text_model_name == 'albert':
            text_outputs = self.albert(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        elif self.text_model_name == 'electra':
            text_outputs = self.electra(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        else:
            text_outputs=None
        if self.image_model_name == 'vit' or self.image_model_name == "vit-large":
            image_outputs = self.vit(pixel_values,head_mask=head_mask)
        elif self.image_model_name == 'swin':
            image_outputs = self.swin(pixel_values,head_mask=head_mask)
        elif self.image_model_name == 'deit':
            image_outputs = self.deit(pixel_values,head_mask=head_mask)
        elif self.image_model_name == 'convnext':
            image_outputs = self.convnext(pixel_values)
        else:
            image_outputs=None

        text_last_hidden_states = text_outputs["last_hidden_state"]            # 16, 60,  768
        image_last_hidden_states = image_outputs["last_hidden_state"]          # 16, 197, 768

        # cross_crf_loss
        image_text_cross_attention, mk, _ = self.image_text_cross(text_last_hidden_states, image_last_hidden_states)
        cross_logits = self.classifier0(image_text_cross_attention)
        mask = (labels != -100)
        mask[:,0] = 1
        cross_crf_loss =  -self.CRF(cross_logits, cross_labels,mask=mask) / 10

        T_e = self.gelu0(F.normalize(self.ys_t(text_last_hidden_states.permute(0, 2, 1)), p=2, dim=-1, eps=1e-7))
        I_e = self.gelu1(F.normalize(self.ys_i(image_last_hidden_states.permute(0, 2, 1)), p=2, dim=-1, eps=1e-7))
        logits_ie = torch.matmul(T_e.permute(0, 2, 1), I_e)
        word_region_align_loss = self.loss_dst(logits_ie, eye(logits_ie))


        # text_loss
        sequence_output1 = self.dropout(self.gelu(self.mlp0(text_last_hidden_states) + mk))
        text_token_logits = self.classifier1(sequence_output1)

        # getTextLoss: CrossEntropy
        text_loss = self.loss_fct(text_token_logits.view(-1, self.text_num_labels), labels.view(-1))
        # text_loss = -self.loss_fct(text_token_logits, cross_labels, mask=mask) / 1000

        loss =  cross_crf_loss + self.beta * word_region_align_loss + self.alpha * text_loss
        
        # end train
        return {"loss":loss,
            "logits":text_token_logits,
            "cross_logits": cross_logits,
                }

def eye(x):
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.float32, device=x.device).unsqueeze(0).expand_as(x)
    return mask

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def distant_cross_entropy(logits, positions, mask=None):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    :param mask: [N]
    '''
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    if mask is not None:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(dtype=log_probs.dtype)))
    else:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss

def cost_matrix_cosine(x, y, eps=1e-5):
    """ Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device
                     ).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(
        b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device
                       ) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2)/beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(txt_emb, img_emb, txt_pad, img_pad,
                           beta=0.5, iteration=50, k=1):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)

    T = ipot(cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad,
             beta, iteration, k)
    distance = trace(cost.matmul(T.detach()))
    return distance





