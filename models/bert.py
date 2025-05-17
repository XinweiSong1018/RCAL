'''
* @name: bert.py
* @description: Functions of BERT-based textual encoders for multimodal tasks.
'''

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

# Expose only the BertTextEncoder class to external modules
__all__ = ['BertTextEncoder']

# Supported transformer models and their tokenizer/model class mappings
TRANSFORMERS_MAP = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
}


class BertTextEncoder(nn.Module):
    """
    Wrapper class for BERT or RoBERTa model that encodes text input into contextualized embeddings.

    Args:
        use_finetune (bool): If True, BERT parameters are updated during training.
        transformers (str): Name of the transformer model to use ('bert' or 'roberta').
        pretrained (str): HuggingFace model name (e.g., 'bert-base-uncased', 'roberta-base').

    Methods:
        get_tokenizer(): Returns the tokenizer associated with the model.
        forward(text): Encodes the input text tensor into contextualized embeddings.
    """
    def __init__(self, use_finetune=False, transformers='bert', pretrained='bert-base-uncased'):
        super().__init__()

        # Select the corresponding tokenizer and model classes
        tokenizer_class = TRANSFORMERS_MAP[transformers][1]
        model_class = TRANSFORMERS_MAP[transformers][0]

        # Load pre-trained tokenizer and model
        self.tokenizer = tokenizer_class.from_pretrained(pretrained)
        self.model = model_class.from_pretrained(pretrained)
        self.use_finetune = use_finetune  # Whether to fine-tune BERT during training

    def get_tokenizer(self):
        """
        Returns the tokenizer for preprocessing raw text into input tensors.
        """
        return self.tokenizer

    def forward(self, text):
        """
        Encodes input text into embeddings using BERT/RoBERTa.

        Args:
            text (Tensor): Tensor of shape (B, 3, L), where:
                - B: batch size
                - 3: input_ids, attention_mask, token_type_ids
                - L: sequence length

        Returns:
            Tensor: Last hidden states of shape (B, L, D), where D is the hidden size (e.g., 768).
        """
        input_ids = text[:, 0, :].long()       # Token indices
        input_mask = text[:, 1, :].float()     # Attention mask
        segment_ids = text[:, 2, :].long()     # Segment (token type) IDs

        if self.use_finetune:
            # Fine-tuning: gradients will be propagated
            last_hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids
            )[0]
        else:
            # Frozen BERT: no gradient updates
            with torch.no_grad():
                last_hidden_states = self.model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids
                )[0]

        return last_hidden_states  # Output shape: (B, L, D)