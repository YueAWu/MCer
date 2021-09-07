"""
Create by Juwei Yue on 2020-3-26
MCer model
"""

from utils import *
from base import *


class MCer(Base):
    def __init__(self, positional_size, n_heads, n_layers, *args, **kwargs):
        super(MCer, self).__init__(*args, **kwargs)
        self.positional_size = positional_size

        self.arg_self_attention = SelfAttention(self.embedding_size, n_heads[0], self.dropout)
        # self.mul_attention = MulAttention(self.embedding_size)
        # self.add_attention = AddAttention(self.embedding_size)
        self.layer_norm = nn.LayerNorm(self.embedding_size)
        self.event_composition = EventComposition(self.embedding_size, self.hidden_size, self.dropout)
        self.gcn = GCN(self.hidden_size, n_layers, self.dropout)
        self.attention = Attention(self.hidden_size)

    def adjust_event_chain_embedding(self, embedding):
        """
        shape: (batch_size, 52, embedding_size) -> (batch_size * 5, 36, embedding_size)
            52: (8 context events + 5 candidate events) * 4 arguments
            36: (8 context events + 1 candidate events) * 4 arguments
        """
        embedding = torch.cat(tuple([embedding[:, i::13, :] for i in range(13)]), 1)
        context_embedding = embedding[:, 0:32, :].repeat(1, 5, 1).view(-1, 32, self.embedding_size)
        candidate_embedding = embedding[:, 32:52, :].contiguous().view(-1, 4, self.embedding_size)
        event_chain_embedding = torch.cat((context_embedding, candidate_embedding), 1)
        return event_chain_embedding

    def adjust_event_embedding(self, embedding):
        """
        shape: (batch_size * 5, 9, hidden_size) -> (batch_size, 13, hidden_size)
        """
        embedding = embedding.view(embedding.size(0) // 5, -1, self.hidden_size)
        context_embedding = torch.zeros(embedding.size(0), 8, self.hidden_size).cuda()
        for i in range(0, 45, 9):
            context_embedding += embedding[:, i:i+8, :]
        context_embedding /= 8.0
        candidate_embedding = embedding[:, 8::9, :]
        event_embedding = torch.cat((context_embedding, candidate_embedding), 1)
        return event_embedding

    def forward(self, inputs, matrix):
        # embedding layer
        inputs_embed = self.embedding(inputs)
        inputs_embed = self.adjust_event_chain_embedding(inputs_embed)

        # argument attention layer
        mask = compute_mask(self.positional_size)
        # 1) self_attention
        arg_embed = self.arg_self_attention(inputs_embed, mask)
        # 2) multiplicative attention
        # arg_embed = self.mul_attention(inputs_embed)
        # 3) additive attention
        # arg_embed = self.add_attention(inputs_embed)
        arg_embed = self.layer_norm(torch.add(inputs_embed, arg_embed))

        # event composition layer
        event_embed = self.event_composition(arg_embed)
        event_embed = self.adjust_event_embedding(event_embed)

        # gcn layer
        gcn_outputs = self.gcn(event_embed, matrix)

        # attention layer
        h_i, h_c = self.attention(gcn_outputs)

        # score functions
        # 1) Euclidean
        outputs = -torch.norm(h_i - h_c, 2, 1).view(-1, 5)
        # 2) Cosine
        # outputs = ((h_i / torch.norm(h_i, dim=1).view(-1, 1)) *
        #            (h_c / torch.norm(h_c, dim=1).view(-1, 1))).sum(-1).view(-1, 5)
        # 3) Dot
        # outputs = (h_i * h_c).sum(-1).view(-1, 5)
        return outputs
