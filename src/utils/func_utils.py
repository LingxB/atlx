import inspect


def filter_params(fn: object, params: dict, override: dict = None) -> dict:
    """Filters `params` and return those in `fn`'s arguments.

    # Arguments
        fn : arbitrary function
        override: dictionary, values to override params

    # Returns
        res : dictionary dictionary containing variables
            in both params and fn's arguments.
    """
    override = override or {}
    res = {}
    fn_args = inspect.getfullargspec(fn)[0]

    for name, value in params.items():
        if name in fn_args:
            res.update({name: value})

    res.update(override)
    return res


