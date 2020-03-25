import os
import argparse
from sklearn.preprocessing import QuantileTransformer

from lib.models import *
from lib.trains import Train
from lib.utils import str2bool
from lib.optim import AdaBound

parser = argparse.ArgumentParser()
# datasets
parser.add_argument('--dataset', default='Biodeg', type=str)
parser.add_argument('--data_dir', default='data/', type=str)
# hyper-parameters for training FNN
parser.add_argument('--ns', type=str2bool, default='n', help='whether use non singleton model')
parser.add_argument('--nns', type=str2bool, default='n', help='whether use nrules non singleton model')

parser.add_argument('--optim', default='adam', type=str)
parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('--n_rules', default=20, type=int)
parser.add_argument('--l2', default=1e-6, type=float)
parser.add_argument('--early_stop', default=30, type=int)
parser.add_argument('--val_size', default=0.1, type=float)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--device', default='cpu', type=str)
# init setup
parser.add_argument('--init', default='fcm', type=str)
parser.add_argument('--scale', default=10, type=float)
parser.add_argument('--nscale', default=10., type=float)
# other exp setup
parser.add_argument('--repeat', default=10, type=int)
args = parser.parse_args()

torch.manual_seed(1447)
np.random.seed(1447)  # setup random seed

# set init func
if args.init == 'kmean':
    init_func = kmean_init
elif args.init == 'fcm':
    init_func = fcm_init
else:
    raise ValueError("Unsupported init function {}".format(args.init))


if 'cls' in args.data_dir:
    classification = True
else:
    classification = False
criterion = torch.nn.CrossEntropyLoss() if classification else torch.nn.MSELoss()

if __name__ == '__main__':
    if not os.path.exists('ckpt'):
        os.mkdir('ckpt')
    if not os.path.exists('ckpt/{}'.format(args.dataset)):
        os.mkdir('ckpt/{}'.format(args.dataset))
    if not os.path.exists('results/{}'.format(args.dataset)):
        os.mkdir('results/{}'.format(args.dataset))

    model_name = 'ns_fnn' if args.ns else 'fnn'
    model_name = 'nns_fnn' if args.nns else model_name

    # load data
    data_file = np.load(os.path.join(args.data_dir, args.dataset + '.npz'))
    if not classification:
        data, label = data_file['data'], data_file['label'].ravel()
        label = label - np.mean(label)
        idx_train, idx_test = data_file['idx_train'], data_file['idx_test']
    else:
        data, label = data_file['con_data'], data_file['label'].ravel()
        idx_train, idx_test = data_file['trains'][0], data_file['tests'][0]
    n_classes = len(np.unique(label)) if classification else 1
    in_dim = data.shape[1]

    x_train, y_train = data[idx_train], label[idx_train]
    x_test, y_test = data[idx_test], label[idx_test]

    pre_pro = QuantileTransformer(output_distribution='normal')
    x_train = pre_pro.fit_transform(x_train)
    x_test = pre_pro.fit_transform(x_test)

    results = np.zeros([4, args.repeat]) if classification else np.zeros([2, args.repeat])
    res_save_path = 'results/{}/{}_lr{}_wd{}_nr{}_{}_s{}_ns{}.npz'.format(
        args.dataset, model_name, args.lr, args.l2, args.n_rules, args.init, args.scale, args.nscale
    )
    print('running to {}'.format(res_save_path))
    if os.path.exists(res_save_path):
        print(res_save_path, "exists")
        exit()

    # begin iteration
    for rep in range(args.repeat):  # repeat $args.repeat times using same train-test validation
        save_path = 'ckpt/{}/{}_rep{}_lr{}_wd{}_nr{}_{}_s{}_ns{}.trained'.format(
            args.dataset, model_name, rep, args.lr, args.l2, args.n_rules, args.init, args.scale, args.nscale
        )

        centers, sigmas = init_func(x_train, n_rules=args.n_rules, scale=args.scale)

        if args.ns:
            ns_simga = np.std(x_train, axis=0) * args.nscale

            if args.nns:
                model = NNonSingletonFNN(
                    in_dim, n_rules=args.n_rules, n_classes=n_classes, init_centers=centers, init_sigmas=sigmas,
                    defuzzy='norm', init_ns_sigma=ns_simga, ravel_out=not classification
                )
            else:
                model = NonSingletonFNN(
                    in_dim, n_rules=args.n_rules, n_classes=n_classes, init_centers=centers, init_sigmas=sigmas,
                    defuzzy='norm', init_ns_sigma=ns_simga, ravel_out=not classification
                )
        else:
            model = FuzzyNeuralNework(
                in_dim, n_rules=args.n_rules, n_classes=n_classes, init_centers=centers, init_sigmas=sigmas,
                defuzzy='norm', ravel_out=not classification
            )

        params_with_l2 = []
        params_without_l2 = []
        for n, p in model.named_parameters():
            if 'weights' in n:  # pytorch default parameter name is weight and bias, but ours: weights and biases
                params_with_l2.append(p)
                print(n)
            else:
                params_without_l2.append(p)
        if args.optim == 'adam':
            optimizer = torch.optim.Adam([
                    {'params': params_without_l2, 'weight_decay': 0},
                    {'params': params_with_l2, 'weight_decay': args.l2}
                ], lr=args.lr)
        elif args.optim == 'adabound':
            optimizer = AdaBound([
                {'params': params_without_l2, 'weight_decay': 0},
                {'params': params_with_l2, 'weight_decay': args.l2}
            ], lr=args.lr)
        else:
            raise ValueError('Unsupport optim type {}'.format(args.optim))

        T = Train(
            model, criterion, optimizer, early_stop=args.early_stop, val_size=args.val_size,
            epochs=args.epochs, batch_size=args.batch_size, device=args.device, save_path=save_path, random_state=1447,
            classification=classification
        )
        if classification:
            valid_acc, valid_bca, test_acc, test_bca = T.fit(x_train, y_train, x_test, y_test)
            print('[{} REP: {}/{}] ACC: {:.4}, BCA: {:.4}'.format(
                args.dataset, rep, args.repeat, test_acc, test_bca
            ))
            results[:, rep] = [valid_acc, valid_bca, test_acc, test_bca]
        else:
            valid_mse, test_mse = T.fit(x_train, y_train, x_test, y_test)
            print('[{} REP: {}/{}] MSE: {:.4}'.format(
                args.dataset, rep, args.repeat, test_mse
            ))
            results[:, rep] = [valid_mse, test_mse]

    np.savez(res_save_path, results=results)
