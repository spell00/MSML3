import torch
import torch.nn as nn
from transformers import BertModel


class LargeSpectralBERT(nn.Module):
    def __init__(self, bert1_config, bert2_config, num_classes):
        super().__init__()
        self.bert1 = BertModel(bert1_config)
        self.bert2 = BertModel(bert2_config)
        self.classifier = nn.Linear(bert2_config.hidden_size, num_classes)

    def forward(self, spectra_patches, attention_mask1=None, attention_mask2=None):
        """
        spectra_patches: [batch_size, n_patches, patch_dim]
        For each spectrum in batch, treat n_patches as BERT1 sequence length.
        """
        batch_size, n_patches, patch_dim = spectra_patches.shape

        # Step 1: For each spectrum, get BERT1 [CLS] embedding
        bert1_cls_embeddings = []
        for i in range(batch_size):
            # [1, n_patches, patch_dim]
            spectrum = spectra_patches[i].unsqueeze(0)
            # Feed as embeddings to BERT1
            bert1_out = self.bert1(inputs_embeds=spectrum, attention_mask=attention_mask1)
            # Take [CLS] token (first token) embedding: [1, hidden]
            cls_emb = bert1_out.last_hidden_state[:, 0, :]  # [1, hidden]
            bert1_cls_embeddings.append(cls_emb)
        # Stack to [batch_size, hidden]
        bert1_cls_embeddings = torch.cat(bert1_cls_embeddings, dim=0)

        # Step 2: Feed sequence of [CLS] embeddings into BERT2
        # For a batch, treat as a sequence: [1, batch_size, hidden]
        bert2_input = bert1_cls_embeddings.unsqueeze(0)  # [1, batch_size, hidden]
        bert2_out = self.bert2(inputs_embeds=bert2_input, attention_mask=attention_mask2)
        # Take [CLS] token of BERT2 (first in sequence)
        final_cls = bert2_out.last_hidden_state[:, 0, :]  # [1, hidden]
        logits = self.classifier(final_cls)
        return logits.squeeze(0) 