#from transformers.models.xlnet import *
from diceloss import MultiDiceLoss
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch
from typing import Optional, List, Tuple
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import SequenceSummary
from attention import SelfAttention
from transformers import XLNetPreTrainedModel, XLNetModel
from focalloss import FocalLoss

class XLNetForTokenClassificationOutput(ModelOutput):
    """
    Output type of :class:`~transformers.XLNetForTokenClassificationOutput`.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class XLNetForTokenClassificationCLS(XLNetPreTrainedModel):
    def __init__(self, config, lossfct=None, CEL_type='mean' ,quick_return=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lossfct = lossfct
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        #self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.CEL_type = CEL_type
        self.quick_return = quick_return
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            use_mems=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where `num_choices` is the size of the second dimension of the input tensors. (see
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state
        '''print(sequence_output)
        print(sequence_output.size())  #torch.Size([6, 10, 1024])'''
        cls_output = self.sequence_summary(sequence_output)
        '''print(cls_output)
        #print(sequence_output[-1])
        print(cls_output.size())#torch.Size([10, 1024])'''
        cls_output = cls_output.unsqueeze(1).repeat(1, outputs[0].size(1), 1)
        '''print(cls_output)
        print(cls_output.size())
        exit()'''
        sequence_output = cls_output + sequence_output
        if self.quick_return:
            return sequence_output
        #sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.lossfct == 'diceloss':
                loss_fct = MultiDiceLoss()
                if attention_mask is not None:
                    '''print(attention_mask)
                    print(attention_mask.shape)  #torch.Size([4, 80])   batch,len'''

                    active_loss = attention_mask.view(-1) == 1
                    '''print(active_loss)
                    print(active_loss.shape)#torch.Size([320])   4*80
                    print(logits)
                    print(logits.shape)  #torch.Size([4, 80, 6])'''
                    active_logits = logits.view(-1, self.num_labels)
                    '''print(active_logits)
                    print(active_logits.shape)#torch.Size([320, 6])  4*80*6'''
                    #active_logits = torch.masked_select(active_logits, (active_loss == 1))
                    active_labels = labels.view(-1)  #->torch.Size([320])
                    active_labels = F.one_hot(active_labels, self.num_labels)
                    '''print(labels)
                    print(labels.shape)#torch.Size([4, 80])
                    print(active_labels)
                    print(active_labels.shape)#torch.Size([320,6])
                    print(active_logits)'''
                    mask = attention_mask.view(-1,1)
                    mask = mask.repeat(1,self.num_labels)
                    '''print(mask)
                    print(mask.shape)#torch.Size([320, 6])'''
                    loss = loss_fct(active_logits, active_labels, mask)
                    #print(loss)
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
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForTokenClassificationOutput(
            loss=loss,
            logits=logits,
            mems=outputs.mems,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class XLNetForTokenClassificationConcatCLSHIDD(XLNetPreTrainedModel):
    def __init__(self, config, lossfct=None, CEL_type='mean' , quick_return=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lossfct = lossfct
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        #self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(5*config.hidden_size, config.num_labels)
        self.CEL_type = CEL_type
        self.quick_return = quick_return
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            #use_mems=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where `num_choices` is the size of the second dimension of the input tensors. (see
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            #use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=None,
        )

        sequence_output = outputs.last_hidden_state
        '''print(sequence_output)
        print(sequence_output.size())  #torch.Size([6, 10, 1024])'''
        cls_output = self.sequence_summary(sequence_output)
        '''print(cls_output)
        #print(sequence_output[-1])
        print(cls_output.size())#torch.Size([10, 1024])'''
        cls_output = cls_output.unsqueeze(1).repeat(1, outputs[0].size(1), 1)
        '''print(cls_output)
        print(cls_output.size())
        exit()'''
        sequence_output = outputs.hidden_states  # tuple(  [batch, len, hiddenstate],)
        sequence_output = torch.cat(sequence_output[-4:], -1)
        sequence_output = torch.cat((sequence_output, cls_output), -1)
        #sequence_output = self.dropout(sequence_output)
        if self.quick_return:
            return sequence_output
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.lossfct == 'diceloss':
                loss_fct = MultiDiceLoss()
                if attention_mask is not None:
                    '''print(attention_mask)
                    print(attention_mask.shape)  #torch.Size([4, 80])   batch,len'''

                    active_loss = attention_mask.view(-1) == 1
                    '''print(active_loss)
                    print(active_loss.shape)#torch.Size([320])   4*80
                    print(logits)
                    print(logits.shape)  #torch.Size([4, 80, 6])'''
                    active_logits = logits.view(-1, self.num_labels)
                    '''print(active_logits)
                    print(active_logits.shape)#torch.Size([320, 6])  4*80*6'''
                    #active_logits = torch.masked_select(active_logits, (active_loss == 1))
                    active_labels = labels.view(-1)  #->torch.Size([320])
                    active_labels = F.one_hot(active_labels, self.num_labels)
                    '''print(labels)
                    print(labels.shape)#torch.Size([4, 80])
                    print(active_labels)
                    print(active_labels.shape)#torch.Size([320,6])
                    print(active_logits)'''
                    mask = attention_mask.view(-1,1)
                    mask = mask.repeat(1,self.num_labels)
                    '''print(mask)
                    print(mask.shape)#torch.Size([320, 6])'''
                    loss = loss_fct(active_logits, active_labels, mask)
                    #print(loss)
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
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForTokenClassificationOutput(
            loss=loss,
            logits=logits,
            mems=outputs.mems,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class XLNetForTokenClassificationConcatCLS(XLNetPreTrainedModel):
    def __init__(self, config, lossfct=None, CEL_type='mean' ,quick_return=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lossfct = lossfct
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        #self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(2*config.hidden_size, config.num_labels)
        self.CEL_type = CEL_type
        self.quick_return = quick_return
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            use_mems=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where `num_choices` is the size of the second dimension of the input tensors. (see
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state
        '''print(sequence_output)
        print(sequence_output.size())  #torch.Size([6, 10, 1024])'''
        cls_output = self.sequence_summary(sequence_output)
        '''print(cls_output)
        #print(sequence_output[-1])
        print(cls_output.size())#torch.Size([10, 1024])'''
        cls_output = cls_output.unsqueeze(1).repeat(1, outputs[0].size(1), 1)
        '''print(cls_output)
        print(cls_output.size())
        exit()'''
        sequence_output = torch.cat((sequence_output, cls_output), -1)
        #sequence_output = self.dropout(sequence_output)
        if self.quick_return:
            return sequence_output
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.lossfct == 'diceloss':
                loss_fct = MultiDiceLoss()
                if attention_mask is not None:
                    '''print(attention_mask)
                    print(attention_mask.shape)  #torch.Size([4, 80])   batch,len'''

                    active_loss = attention_mask.view(-1) == 1
                    '''print(active_loss)
                    print(active_loss.shape)#torch.Size([320])   4*80
                    print(logits)
                    print(logits.shape)  #torch.Size([4, 80, 6])'''
                    active_logits = logits.view(-1, self.num_labels)
                    '''print(active_logits)
                    print(active_logits.shape)#torch.Size([320, 6])  4*80*6'''
                    #active_logits = torch.masked_select(active_logits, (active_loss == 1))
                    active_labels = labels.view(-1)  #->torch.Size([320])
                    active_labels = F.one_hot(active_labels, self.num_labels)
                    '''print(labels)
                    print(labels.shape)#torch.Size([4, 80])
                    print(active_labels)
                    print(active_labels.shape)#torch.Size([320,6])
                    print(active_logits)'''
                    mask = attention_mask.view(-1,1)
                    mask = mask.repeat(1,self.num_labels)
                    '''print(mask)
                    print(mask.shape)#torch.Size([320, 6])'''
                    loss = loss_fct(active_logits, active_labels, mask)
                    #print(loss)
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
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForTokenClassificationOutput(
            loss=loss,
            logits=logits,
            mems=outputs.mems,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class XLNetForTokenClassificationATT(XLNetPreTrainedModel):
    def __init__(self, config, lossfct=None, CEL_type='mean' ,quick_return=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lossfct = lossfct
        self.transformer = XLNetModel(config)
        self.attn = SelfAttention(config.hidden_size)
        self.sequence_summary = SequenceSummary(config)
        #self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.CEL_type = CEL_type
        self.quick_return = quick_return
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            #use_mems=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where `num_choices` is the size of the second dimension of the input tensors. (see
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            #use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        sequence_output = outputs.hidden_states  # tuple(  [batch, len, hiddenstate],)
        layers = len(sequence_output)
        batchsize, length, hidden_size = sequence_output[0].size(0), sequence_output[0].size(1), sequence_output[0].size(2)
        '''print(layers)
        print(batchsize)
        print(length)
        print(hidden_size)'''
        # print(sequence_output.size())
        sequence_output = torch.cat(sequence_output).view(layers, batchsize, length,
                                                          hidden_size)  # tensor.size([layers, batch, len, hiddenstate])
        # print(sequence_output.size())
        sequence_output = sequence_output.transpose(0, 1).transpose(1, 2).contiguous()
        sequence_output = self.attn(sequence_output)

        cls_output = self.sequence_summary(sequence_output)
        '''print(cls_output)
        #print(sequence_output[-1])
        print(cls_output.size())#torch.Size([10, 1024])'''
        cls_output = cls_output.unsqueeze(1).repeat(1, outputs[0].size(1), 1)
        '''print(cls_output)
        print(cls_output.size())
        exit()'''
        sequence_output = cls_output + sequence_output
        if self.quick_return:
            return sequence_output
        #sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.lossfct == 'diceloss':
                loss_fct = MultiDiceLoss()
                if attention_mask is not None:
                    '''print(attention_mask)
                    print(attention_mask.shape)  #torch.Size([4, 80])   batch,len'''

                    active_loss = attention_mask.view(-1) == 1
                    '''print(active_loss)
                    print(active_loss.shape)#torch.Size([320])   4*80
                    print(logits)
                    print(logits.shape)  #torch.Size([4, 80, 6])'''
                    active_logits = logits.view(-1, self.num_labels)
                    '''print(active_logits)
                    print(active_logits.shape)#torch.Size([320, 6])  4*80*6'''
                    #active_logits = torch.masked_select(active_logits, (active_loss == 1))
                    active_labels = labels.view(-1)  #->torch.Size([320])
                    active_labels = F.one_hot(active_labels, self.num_labels)
                    '''print(labels)
                    print(labels.shape)#torch.Size([4, 80])
                    print(active_labels)
                    print(active_labels.shape)#torch.Size([320,6])
                    print(active_logits)'''
                    mask = attention_mask.view(-1,1)
                    mask = mask.repeat(1,self.num_labels)
                    '''print(mask)
                    print(mask.shape)#torch.Size([320, 6])'''
                    loss = loss_fct(active_logits, active_labels, mask)
                    #print(loss)
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
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForTokenClassificationOutput(
            loss=loss,
            logits=logits,
            mems=outputs.mems,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class XLNetForTokenClassificationONATT(XLNetPreTrainedModel):
    def __init__(self, config, lossfct=None, CEL_type='mean' ,quick_return=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lossfct = lossfct
        self.transformer = XLNetModel(config)
        self.attn = SelfAttention(config.hidden_size)
        #self.sequence_summary = SequenceSummary(config)
        #self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.CEL_type = CEL_type
        self.quick_return = quick_return
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            #use_mems=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where `num_choices` is the size of the second dimension of the input tensors. (see
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            #use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        sequence_output = outputs.hidden_states  # tuple(  [batch, len, hiddenstate],)
        layers = len(sequence_output)
        batchsize, length, hidden_size = sequence_output[0].size(0), sequence_output[0].size(1), sequence_output[0].size(2)
        '''print(layers)
        print(batchsize)
        print(length)
        print(hidden_size)'''
        # print(sequence_output.size())
        sequence_output = torch.cat(sequence_output).view(layers, batchsize, length,
                                                          hidden_size)  # tensor.size([layers, batch, len, hiddenstate])
        # print(sequence_output.size())
        sequence_output = sequence_output.transpose(0, 1).transpose(1, 2).contiguous()
        sequence_output = self.attn(sequence_output)
        if self.quick_return:
            return sequence_output
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.lossfct == 'diceloss':
                loss_fct = MultiDiceLoss()
                if attention_mask is not None:
                    '''print(attention_mask)
                    print(attention_mask.shape)  #torch.Size([4, 80])   batch,len'''

                    active_loss = attention_mask.view(-1) == 1
                    '''print(active_loss)
                    print(active_loss.shape)#torch.Size([320])   4*80
                    print(logits)
                    print(logits.shape)  #torch.Size([4, 80, 6])'''
                    active_logits = logits.view(-1, self.num_labels)
                    '''print(active_logits)
                    print(active_logits.shape)#torch.Size([320, 6])  4*80*6'''
                    #active_logits = torch.masked_select(active_logits, (active_loss == 1))
                    active_labels = labels.view(-1)  #->torch.Size([320])
                    active_labels = F.one_hot(active_labels, self.num_labels)
                    '''print(labels)
                    print(labels.shape)#torch.Size([4, 80])
                    print(active_labels)
                    print(active_labels.shape)#torch.Size([320,6])
                    print(active_logits)'''
                    mask = attention_mask.view(-1,1)
                    mask = mask.repeat(1,self.num_labels)
                    '''print(mask)
                    print(mask.shape)#torch.Size([320, 6])'''
                    loss = loss_fct(active_logits, active_labels, mask)
                    #print(loss)
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
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForTokenClassificationOutput(
            loss=loss,
            logits=logits,
            mems=outputs.mems,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
