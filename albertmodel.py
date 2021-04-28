import torch
from torchcrf import CRF
from transformers import AlbertPreTrainedModel, AlbertModel
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    TokenClassifierOutput,
)
import torch.nn.functional as F
from diceloss import MultiDiceLoss
from attention import SelfAttention
from focalloss import FocalLoss
class AlbertForTokenClassificationCLS(AlbertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, lossfct=None, CEL_type='mean' ,quick_return=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lossfct = lossfct
        self.albert = AlbertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.CEL_type = CEL_type
        self.quick_return = quick_return
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        cls_output = outputs[1]

        cls_output = cls_output.unsqueeze(1).repeat(1, outputs[0].size(1), 1)
        sequence_output = cls_output + sequence_output

        if self.quick_return:
            return sequence_output

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.lossfct == 'diceloss':
                loss_fct = MultiDiceLoss()
                if attention_mask is not None:

                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = labels.view(-1)
                    active_labels = F.one_hot(active_labels, self.num_labels)
                    mask = attention_mask.view(-1,1)
                    mask = mask.repeat(1,self.num_labels)
                    loss = loss_fct(active_logits, active_labels, mask)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.lossfct == 'focalloss':
                loss_fct = FocalLoss()  # 'sum'
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(reduction=self.CEL_type)
            # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AlbertForTokenClassificationConcatCLS(AlbertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, lossfct=None, CEL_type='mean' ,quick_return=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lossfct = lossfct
        self.albert = AlbertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2*config.hidden_size, config.num_labels)
        self.CEL_type = CEL_type
        self.quick_return = quick_return
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        cls_output = outputs[1]

        cls_output = cls_output.unsqueeze(1).repeat(1, outputs[0].size(1), 1)
        sequence_output = torch.cat((sequence_output, cls_output), -1)

        if self.quick_return:
            return sequence_output

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.lossfct == 'diceloss':
                loss_fct = MultiDiceLoss()
                if attention_mask is not None:


                    active_loss = attention_mask.view(-1) == 1

                    active_logits = logits.view(-1, self.num_labels)


                    active_labels = labels.view(-1)  #->torch.Size([320])
                    active_labels = F.one_hot(active_labels, self.num_labels)

                    mask = attention_mask.view(-1,1)
                    mask = mask.repeat(1,self.num_labels)

                    loss = loss_fct(active_logits, active_labels, mask)

                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.lossfct == 'focalloss':
                loss_fct = FocalLoss()  # 'sum'
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(reduction=self.CEL_type)
            # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class AlbertForTokenClassificationATT(AlbertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, lossfct=None, CEL_type='mean' ,quick_return=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lossfct = lossfct
        self.albert = AlbertModel(config, add_pooling_layer=True)
        self.attn = SelfAttention(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.CEL_type = CEL_type
        self.quick_return = quick_return
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        sequence_output = outputs[2]
        layers = len(sequence_output)
        batchsize, length, hidden_size = sequence_output[0].size(0), sequence_output[0].size(1), sequence_output[0].size(2)

        sequence_output = torch.cat(sequence_output).view(layers, batchsize, length,
                                                          hidden_size)

        sequence_output = sequence_output.transpose(0, 1).transpose(1, 2).contiguous()
        sequence_output = self.attn(sequence_output)


        cls_output = outputs[1]

        cls_output = cls_output.unsqueeze(1).repeat(1, outputs[0].size(1), 1)
        sequence_output = cls_output + sequence_output

        if self.quick_return:
            return sequence_output

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.lossfct == 'diceloss':
                loss_fct = MultiDiceLoss()
                if attention_mask is not None:


                    active_loss = attention_mask.view(-1) == 1

                    active_logits = logits.view(-1, self.num_labels)

                    active_labels = labels.view(-1)  #->torch.Size([320])
                    active_labels = F.one_hot(active_labels, self.num_labels)

                    mask = attention_mask.view(-1,1)
                    mask = mask.repeat(1,self.num_labels)

                    loss = loss_fct(active_logits, active_labels, mask)

                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.lossfct == 'focalloss':
                loss_fct = FocalLoss()  # 'sum'
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)  # 320*6,  320
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(reduction=self.CEL_type)  #'sum'
            # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)     #320*6,  320
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AlbertForTokenClassificationONATT(AlbertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, lossfct=None, CEL_type='mean' ,quick_return=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lossfct = lossfct
        self.albert = AlbertModel(config)
        self.attn = SelfAttention(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.CEL_type = CEL_type
        self.quick_return = quick_return
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        sequence_output = outputs[2]
        layers = len(sequence_output)
        batchsize, length, hidden_size = sequence_output[0].size(0), sequence_output[0].size(1), sequence_output[0].size(2)

        sequence_output = torch.cat(sequence_output).view(layers, batchsize, length,
                                                          hidden_size)

        sequence_output = sequence_output.transpose(0, 1).transpose(1, 2).contiguous()
        sequence_output = self.attn(sequence_output)
        if self.quick_return:
            return sequence_output

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.lossfct == 'diceloss':
                loss_fct = MultiDiceLoss()
                if attention_mask is not None:


                    active_loss = attention_mask.view(-1) == 1

                    active_logits = logits.view(-1, self.num_labels)

                    active_labels = labels.view(-1)
                    active_labels = F.one_hot(active_labels, self.num_labels)

                    mask = attention_mask.view(-1,1)
                    mask = mask.repeat(1,self.num_labels)

                    loss = loss_fct(active_logits, active_labels, mask)
                    #print(loss)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.lossfct == 'focalloss':
                loss_fct = FocalLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(reduction=self.CEL_type)
            # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
