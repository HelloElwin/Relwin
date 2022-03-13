import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--reg', default=1e-6, type=float, help='weight decay regularizer')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
    parser.add_argument('--tstEpoch', default=3, type=int, help='number of epoch to test while training')
    parser.add_argument('--latdim', default=64, type=int, help='embedding size')
    parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')
    parser.add_argument('--gnn_layer', default=3, type=int, help='number of gnn layers')
    parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')
    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--shoot', default=10, type=int, help='K of top k')
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
    parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
    parser.add_argument('--deep_layer', default=0, type=int, help='number of deep layers to make the final prediction')
    parser.add_argument('--keepRate', default=0.7, type=float, help='rate for dropout')
    parser.add_argument('--slot', default=5, type=float, help='length of time slots')
    parser.add_argument('--graphSampleN', default=15000, type=int, help='use 25000 for training and 200000 for testing, empirically')
    parser.add_argument('--divSize', default=10000, type=int, help='div size for smallTestEpoch')

    parser.add_argument('--ssl_keep_rate', default=0.1, type=float, help='drop out keep rate')
    parser.add_argument('--ssl_temp', default=0.2, type=float, help='InfoNCE temperature')
    parser.add_argument('--ssl_reg', default=1e-6, type=float, help='ssl regularization')

    return parser.parse_args()

# | reg  | ssl_keep_rate | ssl_temp | ssl_reg | HR/NDCG       |
# | ---- | ------------- | -------- | ------- | ------------- |
# | 1e-6 | 0.4           | 0.5      | 1e-8    | 0.7361/0.4511 |
# | 1e-6 | 0.2           | 0.5      | 1e-8    | 0.7457/0.4502 |
# | 1e-6 | 0.4           | 0.5      | 1e-7    | 0.7561/0.4704 |
# | 1e-6 | 0.2           | 0.5      | 1e-7    | 0.7502/0.4650 |
# | 1e-6 | 0.4           | 0.5      | 1e-6    | 0.7646/0.4776 |
# | 1e-6 | 0.2           | 0.5      | 1e-6    | 0.7680/0.4795 |
# | 1e-6 | 0.2           | 0.2      | 1e-6    | 0.7540/0.4646 |
# | 1e-6 | 0.1           | 0.2      | 1e-6    | 0.7686/0.4800 |
# | 1e-6 | 0.2           | 0.5      | 1e-5    | 0.6749/0.3718 |
# | 1e-6 | 0.1           | 0.2      | 1e-5    | 0.6243/0.3405 |

args = parse_args()
args.decay_step = args.trnNum//args.batch
