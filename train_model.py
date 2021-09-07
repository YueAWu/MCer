"""
Create by Juwei Yue on 2019-11-2
Train model
"""

from utils import *
from MCer import MCer
from MCerLSTM import MCerLSTM

MODEL_NAME = sys.argv[1]

POSITIONAL_SIZE = 36  # length of token
EMBEDDING_SIZE = 128  # size of argument
HIDDEN_SIZE = EMBEDDING_SIZE * 4  # size of event
if MODEL_NAME == "MCer":
    N_HEADS = [4]
if MODEL_NAME == "MCerLSTM":
    N_HEADS = [16, 16]
N_LAYERS = 1

DROPOUT = float(sys.argv[2])
MARGIN = float(sys.argv[3])
LR = float(sys.argv[4])
WEIGHT_DECAY = float(sys.argv[5])
MOMENTUM = float(sys.argv[6])

BATCH_SIZE = int(sys.argv[7])
EPOCHS = int(sys.argv[8])
PATIENTS = int(sys.argv[9])

hyper_parameters = {"POSITIONAL_SIZE": POSITIONAL_SIZE,
                    "EMBEDDING_SIZE": EMBEDDING_SIZE,
                    "HIDDEN_SIZE": HIDDEN_SIZE,
                    "N_HEADS": N_HEADS,
                    "N_LAYERS": N_LAYERS,
                    "DROPOUT": DROPOUT,
                    "MARGIN": MARGIN,
                    "LR": LR,
                    "WEIGHT_DECAY": WEIGHT_DECAY,
                    "MOMENTUM": MOMENTUM,
                    "BATCH_SIZE": BATCH_SIZE,
                    "EPOCHS": EPOCHS,
                    "PATIENTS": PATIENTS}

train_set = Data(pickle.load(open("data/metadata/vocab_index_train.data", "rb")))
dev_set = Data(pickle.load(open("data/metadata/vocab_index_dev.data", "rb")))
test_set = Data(pickle.load(open("data/metadata/vocab_index_test.data", "rb")))
word_embedding = get_word_embedding()
dev_index = pickle.load(open("data/metadata/dev_index.pickle", "rb"))
test_index = pickle.load(open("data/metadata/test_index.pickle", "rb"))


def train(model_name):
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
        need_matrix = True
    elif model_name == "MCerLSTM":
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
        need_matrix = False
    else:
        print("Model name error!")
        return
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
                if need_matrix:
                    adj_matrix = train_data[1]
                label = train_data[2]

                model.module.optimizer.zero_grad()
                model.module.train()
                if need_matrix:
                    predict = model(event_chain, adj_matrix)
                else:
                    predict = model(event_chain)
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
                    if need_matrix:
                        val_acc, val_loss, _ = \
                            model.module.predict_eval(dev_data[0], dev_data[1], dev_data[2], dev_index)
                    else:
                        val_acc, val_loss, _ = model.module.predict_eval(dev_data[0], dev_data[2], dev_index)
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
        if need_matrix:
            test_acc, _, test_result = model.module.predict_eval(test_data[0], test_data[1], test_data[2], test_index)
        else:
            test_acc, _, test_result = model.module.predict_eval(test_data[0], test_data[2], test_index)
    print("Test Acc: %f" % test_acc)

    history = [train_acc_history, train_loss_history, val_acc_history, val_loss_history]
    best_result(model_name, model.module.state_dict(), best_val_epoch, best_val_acc, test_acc,
                val_acc_history, test_result, hyper_parameters)


if __name__ == '__main__':
    for i in range(1000):
        start_time = time.time()
        train(MODEL_NAME)
        end_time = time.time()
        print("Run time: %f s" % (end_time - start_time))