from model.lab import Commonality
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script computing commonality series.')
    parser.add_argument('--title_figure', default=0, type=int, help='Title figures.')
    parser.add_argument('--L', default=None, help='Time horizon.', type=str)
    parser.add_argument('--transformation', default=None, help='Time horizon.', type=str)
    parser.add_argument('--commonality_type', default='adjusted_r2', type=str, help='Type of commonality to produce.')
    args = parser.parse_args()
    title_figure = bool(args.title_figure)
    commonality_obj = Commonality(transformation=args.transformation, type_commonality=args.commonality_type,
                                  title_figure=title_figure)
    commonality_obj.commonality()