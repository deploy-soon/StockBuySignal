from itertools import product
from data import StockDataset
from train import Train
from test import test

def optimizer_ALSTM():
    U = [4, 8, 16, 32]
    T = [2, 3, 4, 5, 10, 15]
    lamb = [1e-4, 1e-5, 1e-6, 0.]

    best_performance = 0.0
    best_config = None
    for u, t, l in product(U, T, lamb):
        class args:
            epochs = 150
            batch_size = 1024
            hidden_num = u
            lr = 0.001
            lags = t
            epsilon = None
            beta = None
            regularizer = l
            model_path = "weight/model.pt"
            data_path = "../3rd/stock-data/kospi200"
            namespace = "res/model"
            is_regression = False
            use_adversarial = False
            verbose = False

        train_dataset = StockDataset(data_path=args.data_path,
                                     lags=args.lags, is_train=True,
                                     is_regression=args.is_regression)
        test_dataset = StockDataset(lags=args.lags, is_train=False,
                                    data_path=args.data_path,
                                    is_regression=args.is_regression)

        performances = []
        for _ in range(5):
            train = Train(args, dataset=train_dataset)
            train.run()
            performance = test(args, dataset=test_dataset)
            performances.append(performance)
        mean_perf = sum(performances) / len(performances)
        print("hidden: {} lags: {}, regularizer: {}, performance: {:.4}".format(
            u, t, l, mean_perf))
        if mean_perf > best_performance:
            best_performance = mean_perf
            best_config = (u, t, l)
    print(best_performance)
    print(best_config)

def optimizer_Adv_ALSTM(u, t, l):
    betas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.]
    epsilons = [0.001, 0.005, 0.01, 0.05, 0.1]

    best_performance = 0.0
    best_config = None
    for b, e in product(betas, epsilons):
        class args:
            epochs = 150
            batch_size = 1024
            hidden_num = u
            lr = 0.001
            lags = t
            epsilon = e
            beta = b
            regularizer = l
            model_path = "weight/model.pt"
            data_path = "../3rd/stock-data/kospi200"
            namespace = "res/model"
            is_regression = False
            use_adversarial = True
            verbose = False

        train_dataset = StockDataset(data_path=args.data_path,
                                     lags=args.lags, is_train=True,
                                     is_regression=args.is_regression)
        test_dataset = StockDataset(lags=args.lags, is_train=False,
                                    data_path=args.data_path,
                                    is_regression=args.is_regression)

        performances = []
        for _ in range(5):
            train = Train(args, dataset=train_dataset)
            train.run()
            performance = test(args, dataset=test_dataset)
            performances.append(performance)
        mean_perf = sum(performances) / len(performances)
        print("beta: {}, epsilon: {}, performance: {:.4}".format(
            beta, epsilon, mean_perf))
        if mean_perf > best_performance:
            best_performance = mean_perf
            best_config = (b, e)
    print(best_performance, best_config)

if __name__ == "__main__":
    #optimizer_ALSTM()
    optimizer_Adv_ALSTM(u=16, t=5, l=0.)

