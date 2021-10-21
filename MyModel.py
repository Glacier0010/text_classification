import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, vocab, vec_dim, hidden_dim, output_dim, num_layers, batch_size, fix_size):
        super(MyModel, self).__init__()
        self.batch_size = batch_size
        self.embedding = nn.Embedding(len(vocab), vec_dim)
        self.embedding.weight.data.copy_(vocab.vectors)
        self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(vec_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)
        self.fc = nn.Linear(2*fix_size*hidden_dim, output_dim)
        
    def forward(self, x):
        embedding = self.embedding(x)
        output, hidden_state = self.lstm(embedding)
        output = output.contiguous().view(self.batch_size, -1)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.fc(output)
        return output
    