from comet_ml import Experiment
import torch 
from torch import nn, optim
from preprocess import *
from model import *
from tqdm import tqdm
from scipy.io import loadmat, savemat

hyperparameters = {
    "window_size": 80,
    "rnn_size": 1024,
    "learning_rate": 0.0001,
    "num_epochs": 1,
    "output_size": 7,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, experiment):
    loss_fn = nn.SoftMarginLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

    model.train()
    with experiment.train():
        for e in range(hyperparameters["num_epochs"]):
            for item in tqdm(train_loader):
                inputs = item["inputs"]
                labels = item["y"].to(device)

                optimizer.zero_grad()
                logits = model(inputs)
                loss = loss_fn(logits, labels.squeeze())
                loss.backward()
                optimizer.step()

def test(model, test_loader, experiment):
    totals = {"0":0, "1":0, "2":0, "3":0, "overall":0}
    corrects = {"0":0, "1":0, "2":0, "3":0, "overall":0}

    p = [None, None, None, None]

    save_dir = os.path.join('Results', "p_window_{}.mat".format(hyperparameters['window_size']))
    if os.path.exists(save_dir):
        results = loadmat(save_dir)
        p = [results['p0'], results['p1'], results['p2'], results['p3']]
    
    model.eval()
    with experiment.test():
        with torch.no_grad():
            for item in tqdm(test_loader):
                inputs = item["inputs"]
                labels = item["y"].to(device)

                logits = model(inputs)

                confidence = sum(abs(logits)).item()

                logits[logits <= 0] = -1
                logits[logits >  0] =  1

                num_fakes = torch.sum(labels==1).item()
                num_predicted_fakes = torch.sum(logits==1).item()
                
                if torch.all(torch.eq(logits, labels.squeeze())):
                    corrects["overall"] = corrects["overall"] + 1
                    corrects[str(num_fakes)] = corrects[str(num_fakes)] + 1

                y = np.zeros((2,1))

                if num_fakes == num_predicted_fakes:
                    if num_fakes == 0:
                        y[0] = -1 * confidence
                        y[1] = 0 #TN label
                    elif torch.all(torch.eq(logits, labels.squeeze())):
                        y[0] = confidence
                        y[1] = 1 #TP, detected some number of fakes where there was some
                    else:
                        y[0] = -1 * confidence
                        y[1] = 1 #FN, failed to detect some number of fakes where there was some
                elif num_fakes != 0:
                    y[0] = confidence
                    y[1] = 0 #FP detected some number of fakes, but was wrong number of fakes
                else:
                    y[0] = -1 * confidence
                    y[1] = 1 #FN incorrectly got no fakes
                
                if p[num_fakes] is None:
                    p[num_fakes] = y
                else:
                    p[num_fakes] = np.hstack([p[num_fakes], y])

                totals["overall"] = totals["overall"] + 1
                totals[str(num_fakes)] = totals[str(num_fakes)] + 1
            
        print('Accuracy of network on test windows: %d %%' % (100 * corrects["overall"] / totals["overall"]))
        experiment.log_metric("overall_accuracy", corrects["overall"] / totals["overall"])
        experiment.log_metric("0_accuracy", corrects["0"] / totals["0"])
        experiment.log_metric("1_accuracy", corrects["1"] / totals["1"])
        experiment.log_metric("2_accuracy", corrects["2"] / totals["2"])
        experiment.log_metric("3_accuracy", corrects["3"] / totals["3"])

        if not os.path.exists('Results'):
            os.makedirs('Results')
        p_dict = {'p0': p[0], 'p1':np.hstack([p[1], p[0]]), 'p2':np.hstack([p[2], p[0]]), 'p3':np.hstack([p[3], p[0]])}
        savemat(save_dir, p_dict)

def save(model, PATH):
    print('Saving model...')
    torch.save(model.state_dict(), PATH)

def main():
    experiment = Experiment(log_code=True)
    experiment.log_parameters(hyperparameters)

    train_loader, test_loader, test_IDs = load_dataset(window_size=hyperparameters["window_size"])

    experiment._log_parameter("test_split", test_IDs, 0)

    model = Net(rnn_size=hyperparameters["rnn_size"],
                output_size=hyperparameters["output_size"]
                ).to(device)
    TRAIN = True

    if TRAIN:
        train(model, train_loader, experiment)
        save(model, './model.pt')
    else:
        model.load_state_dict(torch.load('./model.pt'))
        
    test(model, test_loader, experiment)

if __name__ == "__main__":
    main()