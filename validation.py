import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import CNN


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


model = CNN()
model.load_state_dict(torch.load('model.th'))
model.eval()
mnist_test = datasets.MNIST(root='/data', train=False, download=False,
                            transform=transforms.Compose([transforms.ToTensor()]))

data_valid = DataLoader(mnist_test, shuffle=False, num_workers=0)
total = 0
num = 0
for itr, (image, label) in enumerate(data_valid):
    pred = model(image)
    acc = accuracy(pred, label)
    total += acc.detach().numpy()
    num += 1

print('accuracy: ', total / num)
