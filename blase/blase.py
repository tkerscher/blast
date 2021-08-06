#!/usr/bin/env

from argparse import ArgumentParser
from blase import *

def main():
    parser = ArgumentParser(description='Estimates the synchrotron peak of an blazar including a 95% prediction interval given its spectral energy distribution as outputted by the VOUBlazar tool.')
    parser.add_argument('sed', help='Path to input sed.')
    parser.add_argument('-b', '--bag', type=int, default=-1,
     help='Specifies in which bag the sed was during training. Only necessary if the sed was used for training.')
    parser.add_argument('-e', '--no-error', action='store_true',
     help='Outputs only the estimated peak without the error.')
    parser.add_argument('-p', '--precision', type=int, default=2,
     help='Number of digits after the period.')
    args = parser.parse_args()

    estimator = Estimator()
    sed = parse_sed(args.sed)
    peak, err = estimator(bin_data(sed), args.bag)
    
    p = args.precision
    if args.no_error:
        print(f'{peak:.{p}f}')
    else:
        print(f'{peak:.{p}f} (+/- {err:.{p}f})')

if __name__ == '__main__':
    main()
