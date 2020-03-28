import argparse
import sys
from data_load import *
from utils_py import choose_and_cut, get_auc, get_auc_list
from model import *
from sklearn.metrics import precision_recall_curve, roc_curve, auc

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    random_state = args.data_seed
    epochs = args.epochs
    log_interval = 6
    feature_num = 8 #args.feature_num
    data_load_obj, test_set, _, all_set = data_load_kfold(random_state = random_state)
    feature_cut = True

    ensemble_layers = 5
    mlps = []
    mlp_optimizers = []
    train_loaders = []
    val_loaders = []
    for i, (fold_train, fold_val, pos_weight) in enumerate(data_load_obj):
        train_loaders.append(DataLoader(fold_train, batch_size = 512, shuffle = True))
        val_loaders.append(DataLoader(fold_val, batch_size = 1223, shuffle = True))
    for _ in range(ensemble_layers):
        net = MLP(feature_num,16).to(device)
        mlps.append(net)
        mlp_optimizers.append(torch.optim.Adam(net.parameters(), lr = 0.002))
    criterion = nn.BCEWithLogitsLoss() #(pos_weight = torch.FloatTensor([(2869/187)]).to(device) )

    if not args.test:

        best_val_loss = float('inf')
        for i, model in enumerate(mlps):
            best_val_auc = 0
            train_loader = train_loaders[i]
            val_loader = val_loaders[i]
            optimizer = mlp_optimizers[i]
            for epoch in range(epochs):
                model.train()
                avg_loss = 0
                avg_AUC = 0
                for batch_idx, (data,_,labels) in enumerate(train_loader):
                    data,_ = choose_and_cut(data,cut=feature_cut)
                    data = torch.from_numpy(data).float()
                    labels = labels.argmax(axis = 1, keepdim = True).float()
                    data, labels = data.to(device), labels.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output,labels)
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
                print("------------------------")
                print("Train Epoch {} Avg_loss: {:.6f} ".format(epoch+1, avg_loss))
                print("Train Epoch {} Avg_AUC: {:.6f} ".format(epoch+1, avg_AUC))

                with torch.no_grad():
                    model.eval()
                    val_loss = 0
                    for val_data, _, val_labels in val_loader:
                        val_data,_ = choose_and_cut(val_data, cut = feature_cut)
                        val_data = torch.from_numpy(val_data).float()
                        val_labels = val_labels.argmax(axis = 1, keepdim = True).float()
                        val_data, val_labels = val_data.to(device), val_labels.to(device)

                        output = model(val_data)
                        loss = criterion(output, val_labels)
                        val_loss += loss.item()
                        val_auc = get_auc(output, val_labels)
                    val_loss /= len(val_loader)
                    print("Validation Loss: {:.6f}".format(val_loss))
                    print("Validation AUC: {:.6f}".format(val_auc))

                    if best_val_auc < val_auc:
                        best_val_auc = val_auc
                        torch.save(model.state_dict(), 'desat-MLP{}.pt'.format(i+1))
                        print("saved at epoch  %i >>>>>>>>>>>>>>>>>> " %(epoch+1))

    else:
        test_loader = DataLoader(test_set, batch_size =1835, shuffle = True) #True)
        for i in range(ensemble_layers):
            mlps[i].load_state_dict(torch.load('desat-MLP{}.pt'.format(i+1)))
            mlps[i].eval()
        with torch.no_grad():
            avg_AUC = 0
            for test_data, _, test_labels in test_loader:
                test_data,_ = choose_and_cut(test_data, cut = feature_cut)
                test_data = torch.from_numpy(test_data).float()
                test_labels = test_labels.argmax(axis = 1, keepdim = True).float()
                test_data, test_labels = test_data.to(device), test_labels.to(device)
                preds = []
                cri = 10
                for model in mlps:
                    output = model(test_data).sigmoid()
                    preds.append(output.detach().cpu().numpy())
                    x,y, thres = roc_curve(test_labels.cpu(), output.detach().cpu())
                    # print(auc(x,y)) calculate AUC for each MLP
                    if cri > thres[0]: # thres[0]: Always max(score) + 1
                        thresholds = thres
                        cri = thres[0]
                fpr = np.array([])
                tpr = np.array([])
                test_labels = test_labels.cpu().numpy().astype(bool)
                for thres in thresholds:
                    voting = (preds >=thres) # preds shape [ensemble_layers, test_data]
                    voting = np.sum(voting, axis = 0) > (float(ensemble_layers)/2.0)
                    tp = np.sum(voting & test_labels)
                    fp = np.sum((voting ^ test_labels) & ~test_labels)
                    tn = np.sum(~voting & ~test_labels)
                    fn = np.sum((voting ^ test_labels) & test_labels)
                    fpr = np.append(fpr,fp/(fp + tn))
                    tpr = np.append(tpr,tp /(tp + fn))
                #print(fpr)
                #print(tpr)
                score = auc(fpr,tpr)
                print("AUC score: ",score)

                '''
                for model in mlps:
                    output = model(test_data).sigmoid()
                    pred.append(output)
                    precision, recall, thresholds = precision_recall_curve(test_labels.cpu(), output.detach().cpu())
                    print(thresholds[:10])
                    print("----------------------")
                    mean_output1 += (output/float(ensemble_layers))
                    mean_output2 += (output**2/float(ensemble_layers))
                    mean_output3 += (output**4/float(ensemble_layers))



                avg_AUC = get_auc(mean_output1,test_labels)
                avg_AUC2 = get_auc(mean_output2,test_labels)
                avg_AUC3 = get_auc(mean_output3,test_labels)
            '''
            #print("Test Avg_AUC: {:.4f}, {:.4f},{:.4f}, ".format(avg_AUC,avg_AUC2,avg_AUC3))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Desaturation MLP")
    parser.add_argument( '--epochs', type=int, default=350)
    parser.add_argument( '--seed', type = int, default = 123)
    parser.add_argument( '--data_seed', type = int, default = 123)
    parser.add_argument( '--feature_num', type = int, default = 8)
    parser.add_argument( '--test', action= 'store_true' )
    args = parser.parse_args()

    main(args)


