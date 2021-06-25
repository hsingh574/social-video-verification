import torch
import torchvision
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, rnn_size=1024, output_size=7):
        super(Net, self).__init__()

        resnet18 = torchvision.models.resnet18(pretrained=True)

        for param in resnet18.parameters():
            param.requires_grad = False

        self.conv_embedding = nn.Sequential(*list(resnet18.children())[:-2])

        embedding_size = 32768
        self.lstm = nn.LSTM(embedding_size, rnn_size, batch_first=True)
        self.fc = nn.Linear(output_size*rnn_size, output_size)

    def forward(self, inputs):
        embedded_input_streams = []
        for stream in inputs:
            stream = stream.squeeze(0).to(device)
            embedded_input_streams.append(self.conv_embedding(stream).view(stream.size(0), -1))

        cell_states = []
        for embedded_input_stream in embedded_input_streams:
            _, (_, cell_state) = self.lstm(embedded_input_stream.unsqueeze(0))
            cell_states.append(cell_state)

        logits = self.fc(torch.stack(cell_states).view(-1))

        return logits
