import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CountingDataset
from sru import SRU


def main():
    COUNT_MAX_TO = 100
    DEVICE = 'cpu'
    LR = 1e-3
    N_EPOCHS = 200
    BATCH_SIZE = 6

    model = SRU(input_dim=COUNT_MAX_TO, hidden_dim=COUNT_MAX_TO)
    model = model.to(DEVICE)

    # Defining loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Creating Dataset
    train_dataset = CountingDataset(COUNT_MAX_TO)

    train_params = {
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }
    training_loader = DataLoader(train_dataset, **train_params)

    # Training Loop
    for epoch in range(1, N_EPOCHS + 1):
        for _, data in enumerate(training_loader):
            input_seq = data['data'].to(DEVICE)
            target_seq = data['label'].to(DEVICE)

            output = model(input_seq)

            loss = criterion(output[:, 2, :].squeeze(), target_seq.squeeze())   # last digit
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print('epoch: {}, loss: {}'.format(epoch, loss.item()))

    # Evaluation
    test_examples = [[1, 2, 3], [5, 6, 7], [78, 79, 80]]

    for ex in test_examples:
        # Predict on example
        ex_embed = train_dataset.one_hot_encode(ex, COUNT_MAX_TO, len(ex))
        ex_embed = torch.from_numpy(ex_embed).unsqueeze(0)

        output = model(ex_embed)
        predicted_next_digit = output.squeeze()[2, :]
        predicted_next_digit = torch.argmax(predicted_next_digit, -1)

        print('For sequence: {}, next digit is: {}'.format(ex, predicted_next_digit))


if __name__ == '__main__':
    main()
