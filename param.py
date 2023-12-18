import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description="Run Model.")

    parser.add_argument('--data_path', type=str, default='./HMDD_data/MDAv2.0',
                        help='the number of miRANs.')
    parser.add_argument('--validation', type=int, default=5,
                        help='the number of miRANs.')
    parser.add_argument('--epoch', type=int, default=650,
                        help='the number of epoch.')

    parser.add_argument('--mi_num', type=int, default=495, 
                        help='the number of miRANs.')
    parser.add_argument('--dis_num', type=int, default=380,
                        help='the number of diseases.')

    parser.add_argument('--alpha', type=int, default=0.11,
                        help='the size of alpha.')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayer', type=int, default=2,
                        help='Number of layers.')
    parser.add_argument('--n_hidden', type=int, default=20,
                        help='Number of hidden units per modal.')
    parser.add_argument('--n_head', type=int, default=5,
                        help='Number of attention head.')
    parser.add_argument('--nmodal', type=int, default=2,
                        help='Number of views.')

    return parser.parse_args()