# carefully exame
import statistics
import torch
import random
import pickle
import torch.nn as nn
import torch.nn.functional as F


kld = nn.KLDivLoss()


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
