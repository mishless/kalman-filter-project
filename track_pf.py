from tests_pf import test_pf_R, test_pf_Q
from utils import parse_arguments_pf


def main(args):
    if args.test == 'R':
        test_pf_R(args)
    else:
        test_pf_Q(args)


if __name__ == '__main__':
    args = parse_arguments_pf()
    main(args)
