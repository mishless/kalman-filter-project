from utils import parse_arguments
from tests import test_R, test_Q


def main():
    args = parse_arguments()

    if args.test == 'R':
        test_R(args)
    elif args.test == 'Q':
        test_Q(args)


if __name__ == '__main__':
    main()
