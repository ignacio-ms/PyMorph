import sys

from auxiliary.utils.colors import bcolors as c


def arg_check(opt, arg, valid, valid_long, type=None, print_usage=None):
    """
    Check input arguments.
    :param opt: Option.
    :param arg: Argument.
    :param valid: Valid option.
    :param valid_long: Valid long option.
    :param type: Dtype of argument.
    :return: Argument.
    """
    if opt in (valid, valid_long):
        if type is not None:
            try:
                arg = type(arg)
            except ValueError:
                print(f"{c.FAIL}Error{c.ENDC}: {valid_long} must be {type}.")
                if print_usage is not None:
                    print_usage()
                else:
                    sys.exit(2)

        return arg
