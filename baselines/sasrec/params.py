import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--reg', default=1e-6, type=float, help='weight decay regularizer')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs')
    parser.add_argument('--batch', default=128, type=int, help='batch size')
    parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
    parser.add_argument('--latdim', default=64, type=int, help='embedding size')
    parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')
    parser.add_argument('--tstEpoch', default=3, type=int, help='number of epoch to test while training')
    parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')
    parser.add_argument('--shoot', default=10, type=int, help='K for top-K metrics')
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')

    parser.add_argument('--num_his', default=50, type=int, help='max number of history interactions')
    parser.add_argument('--num_layers', default=2, type=int, help='number of attention&feedforward layers')
    parser.add_argument('--num_heads', default=2, type=int, help='number of attention&feedforward layers')
    parser.add_argument('--dropout', default=0, type=float, help='rate of dropout')

    return parser.parse_args()

args = parse_args()
args.decay_step = args.trnNum//args.batch
