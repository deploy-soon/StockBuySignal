from itertools import product
from data import StockDataset
from train import Train
from test import test

def optimizer_ALSTM():
    U = [4, 8, 16, 32]
    T = [2, 3, 4, 5, 10, 15]
    lamb = [0.001, 0.01, 0.1, 1.]

    best_performance = 0.0
    best_config = None
    for u, t, l in product(U, T, lamb):
        class args:
            epochs = 10
            batch_size = 1024
            hidden_num = u
            lr = 0.01
            lags = t
            epsilon = None
            beta = None
            regularizer = l
            model_path = "weight/model.pt"
            namespace = "res/model"
            is_regression = False
            use_adversarial = False
            verbose = False

        train_dataset = StockDataset(lags=args.lags, is_train=True,
                                     is_regression=args.is_regression)
        test_dataset = StockDataset(lags=args.lags, is_train=False,
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
            best_config = args
    print(best_performance)
    print(best_config)

def optimizer_Adv_ALSTM():
    pass

if __name__ == "__main__":
    optimizer_ALSTM()

