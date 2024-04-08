#!/usr/bin/env

from argparse import ArgumentParser
from io import TextIOWrapper
from os import walk
from os.path import isfile, isdir, join
from tqdm import tqdm
from zipfile import ZipFile, is_zipfile
from blast import *

def main():
    parser = ArgumentParser(description='Estimates the synchrotron peak of an blazar including a 95% prediction interval given its spectral energy distribution as outputted by the VOUBlazar tool.')
    parser.add_argument('sed', help='Path to input sed. Can be either a single sed, or a directory or zip file, for which an output file has to be specified.')
    parser.add_argument('-b', '--bag', type=int, default=None,
     help='Specifies in which bag the sed was during training. Overrides automatic detection.')
    parser.add_argument('-e', '--no-error', action='store_true',
     help='Outputs only the estimated peak without the error. Ignored for bulk estimations.')
    parser.add_argument('-f', '--flux', action='store_true',
     help='Estimates flux alongside frequency')
    parser.add_argument('-s', '--silent', action='store_true',
     help='Hides the progressbar for bulk estimation.')
    parser.add_argument('-o', '--out', type=str, default=None,
     help='Output file used if the input is a directory or zip file.')
    parser.add_argument('-p', '--precision', type=int, default=2,
     help='Number of digits after the period. Ignored for bulk estimations.')
    parser.add_argument('-r', '--radius', type=float, default=0.1,
     help='Maximal radius used to search for used training sample.')
    parser.add_argument('-w', '--width', type=float, default=1.96,
     help='Prediction interval width in sigmas (default 1.96 ~ 95%%)')
    args = parser.parse_args()

    p = args.precision
    freq_est = PeakFrequencyEstimator()
    flux_est = PeakFluxEstimator()

    def bulk_estimate(filenames, open_fn):
        if args.out is None:
            print('Bulk estimation issued but no valid output file was specified via -o or --out!')
            return
        with open(args.out, 'w') as out:
            #write header
            cols = [
                'Filename',
                'Right Ascension',
                'Declination',
                'Estimated Peak Frequency',
                'Frequency Error',
            ]
            if args.flux:
                cols.extend(['Estimated Peak Flux','Flux Error'])
            out.write(','.join(cols) + '\n')
            iterator = filenames if args.silent else tqdm(filenames)
            for filename in iterator:
                with open_fn(filename) as file:
                    sed, position = parse_sed(file, position=True)
                    out.write(f'{filename},{position[0]},{position[1]},')
                    if sed is None:
                        print(f'{filename} is not a valid sed and thus skipped')
                        continue
                    bag = get_bag(position, args.radius)
                    peak, err = freq_est(bin_data(sed), bag, sigma=args.width)
                    out.write(f'{peak:.{p}f},{err:.{p}f}')
                    if args.flux:
                        peak, err = flux_est(bin_data(sed), bag, sigma=args.width)
                        out.write(f',{peak:.{p}f},{err:.{p}f}')
                    out.write('\n')

    def print_est(val, err):
        if args.no_error:
            print(f'{val:.{p}f}', end="")
        else:
            print(f'{val:.{p}f} (+/- {err:.{p}f})', end="")

    if isfile(args.sed) and not is_zipfile(args.sed):
        #estimate single sed file
        sed, position = parse_sed(args.sed, position=True)
        if sed is None:
            print('The input file is no valid sed!')
            exit(-1)
        bag = args.bag if args.bag is not None else get_bag(position)
        data = bin_data(sed)
        if args.flux:
            peak, err = flux_est(data, bag, sigma=args.width)
            print_est(peak, err)
            print(" @ ", end="")          
        peak, err = freq_est(bin_data(sed), bag, sigma=args.width)
        print_est(peak, err)
        print("") # new line
    elif is_zipfile(args.sed):
        zipfile = ZipFile(args.sed)
        filenames = [name for name in zipfile.namelist() if name[-1] != '/']
        bulk_estimate(filenames, lambda file: TextIOWrapper(zipfile.open(file)))
    elif isdir(args.sed):
        names = []
        for root, dirnames, filenames in walk(args.sed):
            for filename in filenames:
                names.append(join(root, filename))
        bulk_estimate(names, open)

if __name__ == '__main__':
    main()
