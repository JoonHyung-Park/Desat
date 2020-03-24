import argparse
import sys
from data_load import *
from utils_py import choose_and_cut, get_auc, get_auc_softmax
from model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(2345)
torch.backends.cudnn.deterministic = True

def main(args):

    epochs = args.epochs
    log_interval = 6
    feature_num = 8 #args.feature_num
    train_set, val_set, test_set, _, all_set = data_load_mysplit()
    cls_num_list = [2869,187]
    feature_cut = True

    model = MLP(feature_num,16).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.0008, momentum = 0.9, weight_decay = 2e-4)
    criterion = LDAMLoss(cls_num_list, max_m = 0.5, s=30)
    '''
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    model.apply(init_weights)
    '''
    train_loader = DataLoader(train_set, batch_size = 512, shuffle = True) #True)
    val_loader = DataLoader(val_set,batch_size = 1223, shuffle = True)

    if not args.test:

        best_val_loss = float('inf')
        best_val_auc = 0

        for epoch in range(epochs):

            model.train()
            avg_loss = 0
            avg_AUC = 0
            for batch_idx, (data,_,labels) in enumerate(train_loader):
                data,_ = choose_and_cut(data,cut=feature_cut)
                data = torch.from_numpy(data).float()
                auc_labels = labels
                labels = labels.argmax(axis = 1, keepdim =False ).float()
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output,labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                avg_AUC += get_auc_softmax(output,auc_labels)
                '''
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                '''
            avg_AUC /= len(train_loader)
            avg_loss /= len(train_loader)
            print("------------------------")
            print("Train Epoch {} Avg_loss: {:.6f} ".format(epoch+1, avg_loss))
            print("Train Epoch {} Avg_AUC: {:.6f} ".format(epoch+1, avg_AUC))

            with torch.no_grad():
                model.eval()
                val_loss = 0
                for val_data, _, val_labels in val_loader:
                    val_data,_ = choose_and_cut(val_data, cut = feature_cut)
                    val_data = torch.from_numpy(val_data).float()
                    auc_labels = val_labels
                    val_labels = val_labels.argmax(axis = 1, keepdim = False).float()
                    val_data, val_labels = val_data.to(device), val_labels.to(device)

                    output = model(val_data)
                    loss = criterion(output, val_labels)
                    val_loss += loss.item()
                    val_auc = get_auc_softmax(output, auc_labels)
                val_loss /= len(val_loader)
                print("Validation Loss: {:.6f}".format(val_loss))
                print("Validation AUC: {:.6f}".format(val_auc))

                if best_val_auc < val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), 'desat-MLP.pt')
                    print("saved at epoch  %i >>>>>>>>>>>>>>>>>> " %(epoch+1))

    else:
        test_loader = DataLoader(test_set, batch_size =1835, shuffle = True) #True)
        model.load_state_dict(torch.load('desat-MLP.pt'))
        model.eval()
        with torch.no_grad():
            avg_AUC = 0
            for test_data, _, test_labels in test_loader:
                test_data,_ = choose_and_cut(test_data, cut = feature_cut)
                test_data = torch.from_numpy(test_data).float()
                auc_labels = test_labels
                test_labels = test_labels.argmax(axis = 1, keepdim = False).float()
                test_data, test_labels = test_data.to(device), test_labels.to(device)

                output = model(test_data)
                avg_AUC += get_auc(output,auc_labels)

            avg_AUC /= len(test_loader)
            print("Test Avg_AUC: {:.6f} ".format(avg_AUC))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Desaturation MLP")
    parser.add_argument( '--epochs', type=int, default=1000)
    parser.add_argument( '--feature_num', type = int, default = 8)
    parser.add_argument( '--test', action= 'store_true' )
    args = parser.parse_args()

    main(args)


