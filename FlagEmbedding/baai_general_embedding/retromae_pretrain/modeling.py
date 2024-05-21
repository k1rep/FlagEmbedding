import logging
import os

import torch
from torch import nn
# 用于掩码语言模型的BERT和通用模型
from transformers import BertForMaskedLM, AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

from .arguments import ModelArguments
from .EnhancedDecoder import BertLayerForDecoder

logger = logging.getLogger(__name__)


class RetroMAEForPretraining(nn.Module):
    """
    编码器：BERT作为编码器，处理输入数据并生成隐藏状态。
    解码器：自定义的解码器，利用编码器的输出隐藏状态来进一步处理解码器输入并生成预测。
    编码器-解码器（encoder-decoder）结构
    """
    def __init__(
            self,
            bert: BertForMaskedLM,
            model_args: ModelArguments,
    ):
        super(RetroMAEForPretraining, self).__init__()
        self.lm = bert

        if hasattr(self.lm, 'bert'):
            self.decoder_embeddings = self.lm.bert.embeddings
        elif hasattr(self.lm, 'roberta'):
            self.decoder_embeddings = self.lm.roberta.embeddings
        else:
            self.decoder_embeddings = self.lm.bert.embeddings

        # Decoder Layer Head
        self.c_head = BertLayerForDecoder(bert.config)
        self.c_head.apply(self.lm._init_weights)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args

    def gradient_checkpointing_enable(self, **kwargs):
        """ Enable gradient checkpointing for the model """
        self.lm.gradient_checkpointing_enable(**kwargs)

    def forward(self,
                encoder_input_ids, encoder_attention_mask, encoder_labels,
                decoder_input_ids, decoder_attention_mask, decoder_labels):
        """
        labels 是在训练过程中用于计算损失的目标标签（ground truth）。
        用于掩码语言模型（Masked Language Model, MLM）的目标值
        表示模型在给定输入情况下应该预测的正确词汇
        """
        # encoder part
        # input: encoder_input_ids, encoder_attention_mask, encoder_labels
        # output: hidden_states
        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids, encoder_attention_mask,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True
        )
        cls_hiddens = lm_out.hidden_states[-1][:, :1]  # B 1 D

        # decoder part
        # input: decoder_input_ids, decoder_attention_mask, decoder_labels
        # output: loss
        decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
        hiddens = torch.cat([cls_hiddens, decoder_embedding_output[:, 1:]], dim=1)

        # decoder_position_ids = self.lm.bert.embeddings.position_ids[:, :decoder_input_ids.size(1)]
        # decoder_position_embeddings = self.lm.bert.embeddings.position_embeddings(decoder_position_ids)  # B L D
        # query = decoder_position_embeddings + cls_hiddens

        # classification head
        # hidden states to prediction scores
        cls_hiddens = cls_hiddens.expand(hiddens.size(0), hiddens.size(1), hiddens.size(2))
        query = self.decoder_embeddings(inputs_embeds=cls_hiddens)

        matrix_attention_mask = self.lm.get_extended_attention_mask(
            decoder_attention_mask,
            decoder_attention_mask.shape,
            decoder_attention_mask.device
        )

        hiddens = self.c_head(query=query,
                              key=hiddens,
                              value=hiddens,
                              attention_mask=matrix_attention_mask)[0]
        pred_scores, loss = self.mlm_loss(hiddens, decoder_labels)

        return (loss + lm_out.loss,)

    def mlm_loss(self, hiddens, labels):
        if hasattr(self.lm, 'cls'):
            pred_scores = self.lm.cls(hiddens)
        elif hasattr(self.lm, 'lm_head'):
            pred_scores = self.lm.lm_head(hiddens)
        else:
            raise NotImplementedError

        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return pred_scores, masked_lm_loss

    def save_pretrained(self, output_dir: str):
        # save model weights
        self.lm.save_pretrained(os.path.join(output_dir, "encoder_model"))
        # save state dict
        torch.save(self.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args)
        return model

    def get_input_embeddings(self):
        """
        获取输入的嵌入层表示
        """
        return self.decoder_embeddings.word_embeddings
