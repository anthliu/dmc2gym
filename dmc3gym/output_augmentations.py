import numpy as np

def string_to_oa(s):
    dist, *args = s.split('|')
    args = [float(arg) for arg in args]
    if dist == 'identity':
        return lambda: (lambda u: u)
    elif dist == 'gscale':
        assert len(args) == 2
        def generator():
            coeff = np.random.uniform(args[0], args[1])
            return lambda u: coeff * u
        return generator
    elif dist == 'scale':
        assert len(args) == 3
        def generator():
            coeff = np.random.uniform(args[0], args[1], int(args[2]))
            return lambda u: coeff * u
        return generator
    else:
        raise ValueError(f'Output permutation {s} not implemented')
