import argparse
import sys
from data_load import *
from utils_py import choose_and_cut, get_auc, get_auc_list
from model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True

def main(args):

    epochs = args.epochs
    log_interval = 6
    feature_num = args.feature_num

    # Make Dataloader (k-cross validation)
    data_load_obj, test_set, _, all_set = data_load_mysplit()
    mod = sys.modules[__name__]
    for i, (fold_train, fold_val, pos_weight) in enumerate(data_load_obj):
        setattr(mod, 'train_loader{}'.format(i+1), DataLoader(fold_train, batch_size = 256, shuffle = True))
        setattr(mod, 'val_loader{}'.format(i+1), DataLoader(fold_val, batch_size = 256, shuffle = True))

    test_loader = DataLoader(test_set, batch_size = 256, shuffle = True)

    model = MLP(feature_num,6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.002) #, momentum = 0.9)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        if epoch % 3 == 0 :
            train_loader = train_loader1
            val_loader = val_loader1
        elif epoch % 3 == 1:
            train_loader = train_loader2
            val_loader = val_loader2
        else:
            train_loader = train_loader3
            val_loader = val_loader3

        model.train()
        avg_loss = 0
        avg_AUC = 0
        for batch_idx, (data,_,labels) in enumerate(train_loader):
            data,_ = choose_and_cut(data)
            data = torch.from_numpy(data).float()
            labels = labels.argmax(axis = 1, keepdim = True).float()
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(data)
            #loss = binary_cross_entropy_with_logits(output, labels)
            loss = criterion(output,labels)
            if(epoch == 0):
                pass
                #print("Initial Loss: ", loss.item())
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            avg_AUC += get_auc(output,labels)
            '''
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            '''
        avg_AUC /= len(train_loader)
        avg_loss /= len(train_loader)
        print("---------------------")
        print("Train Epoch {} Avg_loss: {:.6f} ".format(epoch+1, avg_loss))
        print("Train  Epoch {} Avg_AUC: {:.6f} ".format(epoch+1, avg_AUC))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Desaturation MLP")
    parser.add_argument( '--epochs', type=int, default=200)
    #parser.add_argument( '--batch_size', type=int, default=20)
    parser.add_argument( '--feature_num', type = int, default = 8)
    args = parser.parse_args()

    main(args)


