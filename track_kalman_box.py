from utils import parse_arguments_kf
from tests_kf import test_bkf_R, test_bkf_Q


def main():
    args = parse_arguments_kf()

    if args.test == 'R':
        test_bkf_R(args)
    elif args.test == 'Q':
        test_bkf_Q(args)


if __name__ == '__main__':
    main()
