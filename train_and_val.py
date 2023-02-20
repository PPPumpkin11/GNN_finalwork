import torch
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(train_loader, val_loader, epoch_num, optimizer, loss_function, model):
    train_loss_list = []
    val_loss_list = []
    predict_train = []
    predict_validation = []
    # 实现训练+验证
    for epoch in range(epoch_num):
        # scheduler.step()
        train_loss = 0.0
        for idx, (x, y) in enumerate(train_loader):
            # x是(64,8,147)的tensor, y是(64,147)的tensor
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)  # 64*147

            loss = loss_function(output, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if epoch == 99:
                output_cpu = output.cpu()
                predict_train.append(output_cpu.detach().numpy())
        train_loss_list.append(train_loss)
        print('epoch%d--loss on train dataset = %.4f' % (epoch + 1, train_loss))

        if epoch % 9 == 0:
            val_loss = 0.0
            with torch.no_grad():
                for idx, (x, y) in enumerate(val_loader):
                    x = x.to(device)
                    y = y.to(device)
                    output = model(x)
                    loss = loss_function(output, y)
                    val_loss += loss.item()
                    if epoch == 99:
                        output_cpu = output.cpu()
                        predict_validation.append(output_cpu.detach().numpy())
            val_loss_list.append(val_loss)
            print('epoch%d--loss on val dataset=%.4f' % (epoch + 1, val_loss/len(val_loader)))
    return predict_train, predict_validation, train_loss_list, val_loss_list

def show_loss(loss_list):
    x = [i for i in range(len(loss_list))]
    plt.plot(x, loss_list)
    plt.xlabel('batch')
    plt.ylabel('MSE loss')
    plt.show()
