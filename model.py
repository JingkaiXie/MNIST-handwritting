import torch
import torch.nn.functional as F


class CNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7,padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=2),
            torch.nn.ReLU(),

        )
        self.classifier = torch.nn.Linear(8192, 10)

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x.view(-1, 8192))
        x = F.softmax(x)
        return x
