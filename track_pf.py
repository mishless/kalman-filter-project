from tests_pf import test_pf
from utils import parse_arguments_pf


def main(args):
    test_pf(args)


if __name__ == '__main__':
    args = parse_arguments_pf()
    main(args)
