#!/usr/bin/env

from argparse import ArgumentParser
from os import walk
from os.path import isfile, isdir, join
from tqdm import tqdm
from zipfile import ZipFile, is_zipfile
from blase import *

def main():
    parser = ArgumentParser(description='Estimates the synchrotron peak of an blazar including a 95% prediction interval given its spectral energy distribution as outputted by the VOUBlazar tool.')
    parser.add_argument('sed', help='Path to input sed. Can be either a single sed, or a directory or zip file, for which an output file has to be specified.')
    parser.add_argument('-b', '--bag', type=int, default=None,
     help='Specifies in which bag the sed was during training. Overrides automatic detection.')
    parser.add_argument('-e', '--no-error', action='store_true',
     help='Outputs only the estimated peak without the error. Ignored for bulk estimations.')
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
    estimator = Estimator()

    def bulk_estimate(filenames, open_fn):
        if args.out is None:
            print('Bulk estimation issued but no valid output file was specified via -o or --out!')
            return
        with open(args.out, 'w') as out:
            #write header
            out.write('Filename,Right Ascension,Declination,Estimated Peak,Estimation Error (95%)\n')
            iterator = filenames if args.silent else tqdm(filenames)
            for filename in iterator:
                with open_fn(filename) as file:
                    sed, position = parse_sed(file, position=True)
                    if sed is None:
                        print(f'{filename} is not a valid sed and thus skipped')
                        continue
                    bag = get_bag(position, args.radius)
                    peak, err = estimator(bin_data(sed), bag, sigma=args.width)
                    out.write(f'{filename},{position[0]},{position[1]},{peak:.{p}f},{err:.{p}f}\n')


    if isfile(args.sed) and not is_zipfile(args.sed):
        #estimate single sed file
        sed, position = parse_sed(args.sed, position=True)
        if sed is None:
            print('The input file is no valid sed!')
            exit(-1)
        bag = args.bag if args.bag is not None else get_bag(position)
        peak, err = estimator(bin_data(sed), bag, sigma=args.width)
        if args.no_error:
            print(f'{peak:.{p}f}')
        else:
            print(f'{peak:.{p}f} (+/- {err:.{p}f})')
    elif is_zipfile(args.sed):
        zipfile = ZipFile(args.sed)
        filenames = [name for name in zipfile.namelist() if name[-1] != '/']
        bulk_estimate(filenames, zipfile.open)
    elif isdir(args.sed):
        names = []
        for root, dirnames, filenames in walk(args.sed):
            for filename in filenames:
                names.append(join(root, filename))
        bulk_estimate(names, open)

if __name__ == '__main__':
    main()
