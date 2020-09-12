import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from model import CNN
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.tensorboard as tb

def visualize(data):
    fig = plt.figure(figsize=(20, 10))
    for i in range(1, 10):
        img = transforms.ToPILImage(mode='L')(data[i][0])
        fig.add_subplot(1, 10, i)
        plt.title(data[i][1])
        plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    mnist_train = datasets.MNIST(root='/data', train=True, download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))

    data_train = DataLoader(mnist_train, batch_size=256, shuffle=True, num_workers=0)


    # visualize(mnist_test)

    train_logger = tb.SummaryWriter('log/train')
    LR = 0.01
    EPOCHS = 1000
    model = CNN().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,momentum=0.9)
    global_step = 0
    for iterations in range(EPOCHS):
        for image, label in data_train:
            image = Variable(image.cuda())
            label = Variable(label.cuda())
            optimizer.zero_grad()
            output = model(image)
            loss = F.cross_entropy(output, label)
            train_logger.add_scalar('loss', loss, global_step=global_step)
            loss.backward()
            optimizer.step()
            global_step += 1

    torch.save(model.state_dict(), 'model.th')
