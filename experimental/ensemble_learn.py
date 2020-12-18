# learn w for the distill
import statistics
import torch
import random
import pickle
import torch.nn as nn
import torch.nn.functional as F


kld = nn.KLDivLoss()


def fit(x, y, model, opt, loss_fn, epochs=100):
    """Generic function for training a model """
    for epoch in range(epochs):
        loss = loss_fn(model(x), y)

        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item()


class Net(nn.Module):
    def __init__(self, input_hdim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_hdim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)

    def forward(self, input_vecs, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        weight = nn.functional.softmax(x, dim=-1)
        if random.random() < 0.02:
            print(f"W: {weight[0].tolist()}")
        weight = weight.unsqueeze(-1)
        weighted = weight * input_vecs
        pred_dist = torch.sum(weighted, dim=1)
        return pred_dist, weight

# function to calculate loss of a function.
# y_hat -> predicted & y -> actual


def criterion(y_hat, y):
    return kld(
        y_hat, y
    )

# function to calculate accuracy of model


def topk_overlap(distb_a, distb_b, k=5):
    overlap_rate = []
    values_a, indices_a = torch.topk(distb_a, k=k, dim=-1)
    values_b, indices_b = torch.topk(distb_b, k=k, dim=-1)
    indices_b = indices_b.tolist()
    indices_a = indices_a.tolist()
    batch_sz = len(indices_b)
    for b in range(batch_sz):
        overlap = len(set(indices_a[b]).intersection(set(indices_b[b])))
        overlap_rate.append(overlap/k)
    return statistics.mean(overlap_rate)


def test(test_data):
    running_loss = 0
    running_overlap = 0
    cnt = 0
    for i, data in enumerate(test_data):
        # get the inputs; data is a list of [inputs, labels]
        inputs, hidden, meta, labels = data
        with torch.no_grad():
            outputs, weights = net(inputs, hidden)
            loss = -criterion(outputs, labels)
            overlap = topk_overlap(outputs, labels)
            # print statistics
            running_loss += loss.item()
            running_overlap += overlap
            weights = weights.squeeze()
            weights = weights.tolist()
            for idx, w in enumerate(weights):
                m = max(w)
                pos = [i for i, j in enumerate(w) if j == m]

                print(f"{pos}  {meta[idx]['query']}")

            cnt += 1
    print(f"TEST: {running_loss/cnt} . {running_overlap/cnt}")


def prepare_data(input_data, batch_size=20, vocab_size=50264):
    return_data = []
    cursor = 0
    nsamples = len(input_data) // batch_size
    while cursor < nsamples:
        batch_inp = []
        batch_hid = []
        batch_output = []
        batch_meta = []
        for i in range(batch_size):
            data = input_data[cursor*batch_size+i]
            p_lm, p_imp, p_attn, p_full = data['p_lm'], data['p_imp'], data['p_attn'], data['p_full']
            p_lm = p_lm.view(1, -1)[:, :vocab_size]
            p_imp = p_imp.view(1, -1)[:, :vocab_size]
            p_attn = p_attn.view(1, -1)[:, :vocab_size]

            p_full = p_full.view(1, -1)[:vocab_size]

            p_input = torch.cat([p_lm, p_imp, p_attn], dim=0).unsqueeze(0)
            record = data['meta']

            batch_inp.append(p_input)
            batch_hid.append(torch.cat(data['dec_hid']).unsqueeze(0))
            batch_output.append(p_full)
            batch_meta.append(record)
        batch_inp = torch.cat(batch_inp)
        batch_hid = torch.cat(batch_hid)
        batch_output = torch.cat(batch_output)
        return_data.append([batch_inp, batch_hid, batch_meta, batch_output])
        cursor += 1
    return return_data


if __name__ == "__main__":
    fname = 'xsum_run_record_1000.pkl'

    net = Net(6*1024)

    with open(fname, 'rb') as fd:
        data = pickle.load(fd)
    random.seed(2020)
    random.shuffle(data)
    total_num = len(data)

    train_data = data[:int(total_num*0.8)]
    test_data = data[int(total_num*0.8):]
    train_ready_data = prepare_data(train_data)
    test_ready_data = prepare_data(test_data)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(20):  # loop over the dataset multiple times
        print("\n")
        running_loss = 0.0
        running_overlap = 0
        for i, data in enumerate(train_ready_data):
            # get the inputs; data is a list of [inputs, labels]
            inputs, hidden, meta, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = net(inputs, hidden)
            loss = -criterion(outputs, labels) * 1e5
            loss.backward()
            optimizer.step()
            overlap = topk_overlap(outputs, labels)
            # print statistics
            # print(loss.item())
            running_loss += loss.item()
            running_overlap += overlap
            show_step = 20
            if i > 0 and i % show_step == 0:    # print every 2000 mini-batches
                print(
                    f"TRAIN {epoch} - {i} loss: {running_loss/show_step}\t\tOverlap: {running_overlap/show_step}")
                running_loss = 0.0
                running_overlap = 0
        test(test_ready_data)
print('Finished Training')
