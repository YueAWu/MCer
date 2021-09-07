"""
Create by Juwei Yue on 2019-11-2
Based module
"""

from utils import *


class Base(Module):
    def __init__(self, vocab_size, embedding_size, word_embedding, hidden_size,
                 dropout, margin, lr, weight_decay, momentum):
        super(Base, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data = torch.from_numpy(word_embedding)

        self.loss = nn.MultiMarginLoss(margin=margin)
        self.optimizer = optim.RMSprop([{"params": self.get_params()},
                                        {"params": self.embedding.parameters(), "lr": lr*0.06}],
                                       lr=lr, weight_decay=weight_decay, momentum=momentum)

    def get_params(self):
        model_grad_params = filter(lambda p: p.requires_grad, self.parameters())
        train_params = list(map(id, self.embedding.parameters()))
        tune_params = filter(lambda p: id(p) not in train_params, model_grad_params)
        return tune_params

    @staticmethod
    def predict(predict, label):
        _, predict = torch.sort(predict, descending=True)
        n_correct = torch.sum((predict[:, 0] == label)).item()
        n_label = label.size(0)
        acc = n_correct / n_label * 100.0
        return acc

    def predict_eval(self, inputs, matrix, label, set_index):
        predict = self.forward(inputs, matrix)
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


class InitLinear(Module):
    """
    Initialize Linear layer to be distribution function
    """
    def __init__(self, inputs_size, outputs_size, dis_func, func_value, bias=True):
        super(InitLinear, self).__init__()
        self.outputs_size = outputs_size

        self.weight = Parameter(torch.empty(inputs_size, outputs_size), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty(outputs_size), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters(dis_func, func_value)

    def reset_parameters(self, dis_func, func_value):
        if dis_func == "uniform":
            nn.init.uniform_(self.weight, -func_value, func_value)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -func_value, func_value)

        if dis_func == "normal":
            nn.init.normal_(self.weight, std=func_value)
            if self.bias is not None:
                nn.init.normal_(self.bias, std=func_value)

    def forward(self, inputs):
        output_size = inputs.size()[:-1] + (self.outputs_size,)
        if self.bias is not None:
            outputs = torch.addmm(self.bias, inputs.view(-1, inputs.size(-1)), self.weight)
        else:
            outputs = torch.mm(inputs.view(-1, inputs.size(-1)), self.weight)
        outputs = outputs.view(*output_size)
        return outputs


class SelfAttention(Module):
    """
    Self-Attention Layer

    Inputs:
        inputs: word embedding
        inputs.shape = (batch_size, sequence_length, embedding_size)

    Outputs:
        outputs: word embedding with context information
        outputs.shape = (batch_size, sequence_length, embedding_size)
    """
    def __init__(self, embedding_size, n_heads, dropout):
        super(SelfAttention, self).__init__()
        assert embedding_size % n_heads == 0
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.d_k = embedding_size // n_heads

        self.w_qkv = InitLinear(embedding_size, embedding_size*3, dis_func="normal", func_value=0.02)
        self.w_head = InitLinear(embedding_size, embedding_size, dis_func="normal", func_value=0.02)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, k=False):
        x = x.view(-1, x.size(1), self.n_heads, self.d_k)
        if k:
            return x.permute(0, 2, 3, 1)  # key.shape = (batch_size, n_heads, d_k, sequence_length)
        else:
            return x.permute(0, 2, 1, 3)  # query, value.shape = (batch_size, n_heads, sequence_length, d_k)

    def attention(self, query, key, value, mask=None):
        att = torch.matmul(query, key) / math.sqrt(self.d_k)

        if mask is not None:
            att = att + mask

        att = torch.softmax(att, -1)
        att = self.dropout(att)

        # att = self.sample(att)

        outputs = torch.matmul(att, value)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(-1, x.size(1), x.size(2) * x.size(3))
        x = self.dropout(self.w_head(x))
        return x

    @staticmethod
    def sample(att):
        att_ = att.view(-1, att.size(-1))
        _, tk = torch.topk(att_, k=26, largest=False)
        for i in range(att_.size(0)):
            att_[i][tk[i]] = 0.0
        att = att_.view(att.size())
        return att

    def forward(self, inputs, mask=None):
        inputs = self.w_qkv(inputs)
        query, key, value = torch.split(inputs, self.embedding_size, 2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        att_outputs = self.attention(query, key, value, mask)
        outputs = self.merge_heads(att_outputs)
        return outputs


class EventComposition(Module):
    """
    Event Composition layer
        integrate event argument embedding into event embedding

    Inputs:
        inputs: arguments embedding
        inputs.shape = (batch_size, argument_length, embedding_size)

    Outputs:
        outputs: event embedding
        outputs.shape = (batch_size, event_length, hidden_size)
    """
    def __init__(self, inputs_size, outputs_size, dropout):
        super(EventComposition, self).__init__()
        self.outputs_size = outputs_size

        self.w_e = InitLinear(inputs_size*4, outputs_size, dis_func="normal", func_value=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = inputs.view(-1, inputs.size(1) // 4, self.outputs_size)
        outputs = self.dropout(torch.tanh(self.w_e(inputs)))
        return outputs


class GCN(Module):
    def __init__(self, hidden_size, n_layers, dropout, in_out=False):
        super(GCN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.in_out = in_out

        self.linear_list = nn.ModuleList([InitLinear(hidden_size, hidden_size, dis_func="normal", func_value=0.02)
                                          for _ in range(n_layers)])
        if in_out:
            self.linear = InitLinear(hidden_size*2, hidden_size, dis_func="normal", func_value=0.02)

    @staticmethod
    def normalize(a, x=None, in_out=False):
        if x is not None:
            n = torch.tanh(torch.matmul(x, x.permute(0, 2, 1)))
            a = a + torch.tanh(a * n)
            return a

        a = torch.add(a, torch.eye(a.size(1)).cuda())
        d = torch.zeros_like(a)

        if in_out:
            d_in = torch.zeros_like(a)
            d_out = torch.zeros_like(a)
            d_i = torch.sum(a, 1)
            d_o = torch.sum(a, 2)
            for i in range(a.size(0)):
                d_in[i] = torch.diag(torch.pow(d_i[i], -0.5))
                d_out[i] = torch.diag(torch.pow(d_o[i], -0.5))
            return [torch.matmul(torch.matmul(d_in, a), d_in), torch.matmul(torch.matmul(d_out, a), d_out)]

        di = torch.sum(a, 2)
        for i in range(a.size(0)):
            d[i] = torch.diag(torch.pow(di[i], -0.5))
        return torch.matmul(torch.matmul(d, a), d)

    @staticmethod
    def masked(l, reverse=False):
        if not reverse:
            return torch.triu(l, 0)
        else:
            return torch.tril(l, 0)

    def forward(self, inputs, matrix):
        l = self.normalize(matrix, in_out=self.in_out)
        h = inputs

        if self.in_out:
            h_in = h
            h_out = h
            for linear in self.linear_list:
                h = torch.relu(linear(torch.matmul(l[0] + l[1], h)))
                h_in = torch.relu(linear(torch.matmul(l[0], h_in)))
                h_out = torch.relu(linear(torch.matmul(l[1], h_out)))
            h = self.linear(torch.cat([h_in, h_out], 2))
            return h

        for linear in self.linear_list:
            h = torch.relu(linear(torch.matmul(l, h)))
            h = nn.Dropout(self.dropout)(h)
        return h


class Attention(Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.w_i1 = nn.Linear(hidden_size, hidden_size // 2)
        self.w_i2 = nn.Linear(hidden_size // 2, 1)
        self.w_c1 = nn.Linear(hidden_size, hidden_size // 2)
        self.w_c2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, inputs):
        context = inputs[:, 0:8, :].repeat(1, 5, 1).view(-1, 8, self.hidden_size)
        candidate = inputs[:, 8:13, :]
        s_i = torch.relu(self.w_i1(context))
        s_i = torch.relu(self.w_i2(s_i))
        s_c = torch.relu(self.w_c1(candidate))
        s_c = torch.relu(self.w_c2(s_c))
        u = torch.tanh(torch.add(s_i.view(-1, 8), s_c.view(-1, 1)))
        a = (torch.exp(u) / torch.sum(torch.exp(u), 1).view(-1, 1)).view(-1, 8, 1)
        h_i = torch.sum(torch.mul(context, a), 1)
        h_c = (candidate / 8.0).view(-1, self.hidden_size)
        return h_i, h_c


class MulAttention(Module):
    def __init__(self, hidden_size):
        super(MulAttention, self).__init__()

        self.w = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs, mask=None):
        inputs_t = inputs.permute(0, 2, 1)
        u = torch.matmul(self.w(inputs), inputs_t)
        a = torch.softmax(u, 2)
        if mask is not None:
            a = a + mask
        h = torch.matmul(a, inputs)
        return h


class AddAttention(Module):
    def __init__(self, hidden_size):
        super(AddAttention, self).__init__()

        self.hidden_size = hidden_size

        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.q = Parameter(torch.empty(hidden_size), requires_grad=True)
        nn.init.normal_(self.q, -0.02, 0.02)

    def forward(self, inputs, mask=None):
        arg_len = inputs.size(1)

        u1 = self.w1(inputs)
        u2 = self.w2(inputs)
        u = torch.zeros(inputs.size(0), arg_len, arg_len).cuda()
        for i in range(1, arg_len+1):
            u[:, i-1:i, :] = torch.matmul(torch.tanh(u1[:, i-1:i, :] + u2), self.q).view(-1, 1, arg_len)
        a = torch.softmax(u, 2)
        if mask is not None:
            a = a + mask
        h = torch.matmul(a, inputs)
        return h


def compute_mask(positional_size):
    """
    Compute Mask matrix
    Mask: upper triangular matrix of masking subsequent information
        mask value: -1e9
    shape: (positional_size, positional_size)
    """
    return torch.triu(torch.fill_(torch.zeros(positional_size, positional_size), -1e9), 1).cuda()

def train(model_name):
    train_set = Data(pickle.load(open("data/metadata/vocab_index_train.data", "rb")))
    dev_set = Data(pickle.load(open("data/metadata/vocab_index_dev.data", "rb")))
    test_set = Data(pickle.load(open("data/metadata/vocab_index_test.data", "rb")))
    word_embedding = get_word_embedding()
    dev_index = pickle.load(open("data/metadata/dev_index.pickle", "rb"))
    test_index = pickle.load(open("data/metadata/test_index.pickle", "rb"))

    if model_name == "MCer":
        model = MCer(positional_size=POSITIONAL_SIZE,
                     vocab_size=len(word_embedding),
                     embedding_size=EMBEDDING_SIZE,
                     word_embedding=word_embedding,
                     hidden_size=HIDDEN_SIZE,
                     n_heads=N_HEADS,
                     n_layers=N_LAYERS,
                     dropout=DROPOUT,
                     margin=MARGIN,
                     lr=LR,
                     weight_decay=WEIGHT_DECAY,
                     momentum=MOMENTUM)
    if model_name == "MCerLSTM":
        model = MCerLSTM(positional_size=POSITIONAL_SIZE,
                         vocab_size=len(word_embedding),
                         embedding_size=EMBEDDING_SIZE,
                         word_embedding=word_embedding,
                         hidden_size=HIDDEN_SIZE,
                         n_heads=N_HEADS,
                         n_layers=N_LAYERS,
                         dropout=DROPOUT,
                         margin=MARGIN,
                         lr=LR,
                         weight_decay=WEIGHT_DECAY,
                         momentum=MOMENTUM)
    model = to_cuda(nn.DataParallel(model, device_ids=[0, 1]))

    n_train_set = len(train_set.label)
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []
    best_val_acc = 0.0
    best_val_epoch = 0
    start = time.time()
    n_cycle = 0
    while True:
        patient = 0
        epoch = 0
        for epoch in range(EPOCHS):
            print("Epoch %d: " % (EPOCHS * n_cycle + epoch + 1))
            n_batch = BATCH_SIZE
            while train_set.flag_epoch:
                train_data = train_set.next_batch(BATCH_SIZE)
                event_chain = train_data[0]
                adj_matrix = train_data[1]
                label = train_data[2]

                model.module.optimizer.zero_grad()
                model.module.train()
                predict = model(event_chain, adj_matrix)
                loss = model.module.loss(predict, label)
                loss.backward()
                model.module.optimizer.step()

                train_acc = model.module.predict(predict, label)
                train_acc_history.append((time.time() - start, EPOCHS * n_cycle + epoch + 1, train_acc))
                train_loss = loss.item()
                train_loss_history.append((time.time() - start, EPOCHS * n_cycle + epoch + 1, train_loss))

                dev_data = dev_set.all_data()
                model.module.eval()
                with torch.no_grad():
                    val_acc, val_loss, _ = model.module.predict_eval(dev_data[0], dev_data[1], dev_data[2],
                                                                     dev_index)
                val_acc_history.append((time.time() - start, EPOCHS * n_cycle + epoch + 1, val_acc))
                val_loss_history.append((time.time() - start, EPOCHS * n_cycle + epoch + 1, val_loss))
                print("[%6d/%d]:  Train Acc: %f,  Train Loss: %f,  Val Acc: %f,  Val Loss: %f"
                      % (n_batch, n_train_set, train_acc, train_loss, val_acc, val_loss))

                n_batch += BATCH_SIZE
                if n_batch >= n_train_set:
                    n_batch = n_train_set
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_val_epoch = EPOCHS * n_cycle + epoch + 1
                    patient = 0
                else:
                    patient += 1
                if patient > PATIENTS:
                    break
            train_set.flag_epoch = True
            if patient > PATIENTS:
                break
        if epoch == EPOCHS - 1:
            n_cycle += 1
            continue
        else:
            break
    print("Epoch %d: Best Acc: %f" % (best_val_epoch, best_val_acc))
    test_data = test_set.all_data()
    model.module.eval()
    with torch.no_grad():
        test_acc, _, test_result = model.module.predict_eval(test_data[0], test_data[1], test_data[2], test_index)
    print("Test Acc: %f" % test_acc)

    history = [train_acc_history, train_loss_history, val_acc_history, val_loss_history]
    tmp_result(model_name, best_val_epoch, best_val_acc, test_acc, history, hyper_parameters,
               {"DROPOUT": DROPOUT, "MARGIN": MARGIN, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY, "MOMENTUM": MOMENTUM,
                "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS, "PATIENTS": PATIENTS})
    best_result(model_name, model.module.state_dict(), best_val_epoch, best_val_acc, test_acc,
                val_acc_history, test_result, hyper_parameters)