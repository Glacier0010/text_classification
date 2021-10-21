import torch
import torch.nn as nn

from torchtext.legacy.data import Field, Iterator, BucketIterator, TabularDataset

from MyModel import MyModel

        
def load_data(fix_length=100, batch_size=10):
    TEXT = Field(sequential=True, lower=True, fix_length=fix_length, batch_first=True)
    LABEL = Field(sequential=False, use_vocab=False)
    
    train_fields = [('label', LABEL), ('title', None), ('text', TEXT)]
    test_fields = [('label', LABEL), ('title', None), ('text', TEXT)]
    
    train = TabularDataset(path='./dataset/train.csv', format='csv', fields=train_fields, skip_header=True)
    train_iter = BucketIterator(train, batch_size=batch_size, device=-1, sort_key=lambda x: len(x.text), sort_within_batch=False, repeat=False)
    
    test = TabularDataset(path='./dataset/test.csv', format='csv', fields=test_fields, skip_header=True)
    test_iter = Iterator(test, batch_size=batch_size, device=-1, sort=False, sort_within_batch=False, repeat=False)
    
    TEXT.build_vocab(train, vectors='glove.6B.100d')
    vocab = TEXT.vocab
    return train_iter, test_iter, vocab
    
def train(train_iter, model, epochs=5, lr=0.001):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for e in range(epochs):
        for batch_i, content in enumerate(train_iter):
            text, target = content.text, content.label - 1          #label: [0,3]            
            output = model(text)
            
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print("Epoch: {0} \t Batch: {1} \t Loss: {2:.8f}".format(e+1, batch_i, loss))

def test(test_iter, model):
    model.eval()
    total_acc, total_count = 0, 0
    
    with torch.no_grad():
        for batch_i, content in enumerate(train_iter):
            text, target = content.text, content.label - 1          #label: [0,3]            
            output = model(text)
            _, predicted = torch.max(output.data, 1)
            
            total_count += target.size(0)
            total_acc += (predicted == target).sum().item()
        
    total_acc /= total_count
    print('Final accuracy={.4f}', total_acc)
    
    
if __name__ == '__main__':
    fix_length = 200
    batch_size = 100
    train_iter, test_iter, vocab = load_data(fix_length, batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = MyModel(vocab=vocab,
                    vec_dim=200,    #dim of word vector
                    hidden_dim=150,
                    output_dim=4,
                    num_layers=2,
                    batch_size=batch_size,
                    fix_size=fix_length
                    ).to(device)   
    
    train(train_iter, 
          model,
          epochs=1,
          lr=0.001)
    
    test(test_iter, model)