import argparse
import ast

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')

    parser.add_argument("--path", type=str, default="../../../../../../vol/aimspace/users/wyo/registered_meshes/2000/")
    parser.add_argument("--organ", type=str, default="liver_mesh.ply")
    parser.add_argument("--save", type=ast.literal_eval, default=False)

    
    parser.add_argument("--label", type=str, default="VAT")
    parser.add_argument('--epochs', dest='epochs', type=int,
            help='Epochs.', default=30)
    parser.add_argument('--log_interval', dest='log_interval', type=int,
            help='Epochs.', default=5)
    parser.add_argument('--batchs', dest='batchs', type=int,
            help='Batches.', default=64)
    
    parser.add_argument('--num_gc_layers', dest='num_gc_layers', type=int, default=5,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=64,
            help='')
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", type=str, default="StepLR")

    parser.add_argument('--step_size', dest='step_size', type=float,
            help='step_size.', default=1.00)
    parser.add_argument('--gamma', dest='gamma', type=float,
            help='gamma.', default=0.7)

    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.', default=0.001)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
            help='Learning rate.', default=0.00)
    parser.add_argument('--momentum', dest='momentum', type=float,
            help='Learning rate.', default=0.00)

    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=True)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=True)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=True)
    parser.add_argument("--mlp", type=ast.literal_eval, default=True)

    return parser.parse_args()

