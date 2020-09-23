import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import numpy as np
#参考 https://wmathor.com/index.php/archives/1407/
class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
    def forward(self, x):
        #x.shape = (batch_size, 28, 28, 1)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        q = self.encoder(x)#(batch_size, 32)
        mu, sigma = q.chunk(2, dim=1)#(batch_size, 16) (batch_size, 16)

        q = mu + sigma * torch.rand_like(sigma)

        x_hat = self.decoder(q)
        x_hat = x.view(batch_size, 28, 28, 1)

        kld = torch.mean(
            torch.square(mu)+torch.square(sigma)
            - torch.log(1e-8+torch.square(sigma)) - 1
            )
        return x_hat, kld


def get_mnist():
    current_file = os.path.realpath(__file__)
    current_path = os.path.split(current_file)[0]
    data_file = os.path.join(current_path, '../data/mnist/mnist.npz')
    data = np.load(data_file)
    #print(data.files)
    x, y = data['x'], data['y']
    #print(x.shape, y.shape)
    return x, y

def test_VariationalAutoEncoder():
    import matplotlib.pyplot as plt
    plt.ion()
    

    x, y = get_mnist()
    x = x/255.0
    x_train = x[0:60000]
    x_test = x[60000:]
    y_train = y[0:60000]
    y_test = y[60000:]


    train_dataset = data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataset = data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
    test_dataloader = data.DataLoader(test_dataset, batch_size=4, shuffle=True)

    epochs = 50
    lr = 1e-3
    model = VariationalAutoEncoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    fig = plt.figure()
    ax = fig.subplots(2, 4)

    for epoch in range(epochs):
        for batch_index, (train_x, train_y) in enumerate(train_dataloader):
            x_hat, kld = model(train_x)
            loss = criterion(x_hat, train_x) + kld
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            print(epoch, 'loss:', loss.item())
        test_x, _ = iter(test_dataloader).next()
        with torch.no_grad():
            x_hat, _ = model(test_x)

        for i in range(4):
            ax[0, i].imshow(x_hat[i].reshape(28, 28))
            ax[1, i].imshow(test_x[i].reshape(28, 28))
        plt.pause(0.1)

if __name__ == '__main__':
    test_VariationalAutoEncoder()

