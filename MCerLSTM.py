"""
Create by Juwei Yue on 2020-4-7
MCerLSTM model
"""

from utils import *
from base import *


class MCerLSTM(Module):
    def __init__(self, positional_size, vocab_size, embedding_size, word_embedding, hidden_size, n_heads, n_layers,
                 dropout, margin, lr, weight_decay, momentum):
        super(MCerLSTM, self).__init__()
        self.positional_size = positional_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data = torch.from_numpy(word_embedding)

        self.arg_self_attention = SelfAttention(embedding_size, n_heads[0], dropout)
        self.layer_norm_a = nn.LayerNorm(embedding_size)
        self.event_composition = EventComposition(embedding_size, hidden_size, dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.event_self_attention = SelfAttention(hidden_size, n_heads[1], dropout)
        self.layer_norm_e = nn.LayerNorm(hidden_size)
        self.linear = InitLinear(hidden_size*2, 1, dis_func="normal", func_value=0.02)

        self.loss = nn.MultiMarginLoss(margin=margin)
        self.optimizer = optim.RMSprop([{"params": self.get_params()},
                                        {"params": self.embedding.parameters(), "lr": lr*0.06}],
                                       lr=lr, weight_decay=weight_decay, momentum=momentum)

    def get_params(self):
        model_grad_params = filter(lambda p: p.requires_grad, self.parameters())
        train_params = list(map(id, self.embedding.parameters()))
        tune_params = filter(lambda p: id(p) not in train_params, model_grad_params)
        return tune_params

    def adjust_event_chain_embedding(self, embedding):
        """
        shape: (batch_size, 52, embedding_size) -> (batch_size * 5, 36, embedding_size)
            52: (8 context_event + 5 candidate event) * 4 arguments
            36: (8 context_event + 1 candidate event) * 4 arguments
        """
        embedding = torch.cat(tuple([embedding[:, i::13, :] for i in range(13)]), 1)
        context_embedding = embedding[:, 0:32, :].repeat(1, 5, 1).view(-1, 32, self.embedding_size)
        candidate_embedding = embedding[:, 32:52, :].contiguous().view(-1, 4, self.embedding_size)
        event_chain_embedding = torch.cat((context_embedding, candidate_embedding), 1)
        return event_chain_embedding

    def forward(self, inputs):
        self.lstm.flatten_parameters()

        # embedding layer
        inputs_embed = self.embedding(inputs)
        inputs_embed = self.adjust_event_chain_embedding(inputs_embed)

        # argument self-attention layer
        mask_a = compute_mask(self.positional_size)
        arg_embed = self.arg_self_attention(inputs_embed, mask_a)
        arg_embed = self.layer_norm_a(torch.add(inputs_embed, arg_embed))

        # event composition layer
        event_embed = self.event_composition(arg_embed)

        # LSTM layer
        lstm_inputs = event_embed.transpose(0, 1)
        lstm_outputs, (_, _) = self.lstm(lstm_inputs)
        lstm_outputs = lstm_outputs.transpose(0, 1).contiguous()
        decoder_embed = lstm_outputs[:, -1:, :].squeeze()

        # event self-attention layer
        event_embed = self.event_self_attention(event_embed)
        event_embed = self.layer_norm_e(torch.add(lstm_outputs, event_embed))
        encoder_embed = event_embed[:, -1:, :].squeeze()

        # linear layer
        outputs = torch.cat([encoder_embed, decoder_embed], 1)
        outputs = self.linear(outputs)
        outputs = outputs.view(-1, 5)
        return outputs

    @staticmethod
    def predict(predict, label):
        _, predict = torch.sort(predict, descending=True)
        n_correct = torch.sum((predict[:, 0] == label)).item()
        n_label = label.size(0)
        acc = n_correct / n_label * 100.0
        return acc

    def predict_eval(self, inputs, label, set_index):
        predict = self.forward(inputs)
        loss = self.loss(predict, label)
        for index in set_index:
            predict[index] = -1e9
        _, predict = torch.sort(predict, descending=True)
        n_correct = torch.sum((predict[:, 0] == label)).item()
        acc = n_correct / label.size(0) * 100.0
        predict = predict[:, 0:1].squeeze().cpu().numpy().tolist()
        label = label.cpu().numpy().tolist()
        predict_result = []
        for i in range(len(label)):
            if predict[i] == label[i]:
                predict_result.append(1)
            else:
                predict_result.append(0)
        return acc, loss, predict_result
