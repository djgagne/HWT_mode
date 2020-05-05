import argparse
from os.path import join, exists
from os import makedirs
from glob import glob
import pandas as pd
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", default="today", help="First date to extract")
    parser.add_argument("-e", "--end", default="today", help="Last date to extract")
    parser.add_argument("-m", "--mem", default="mem_1", help="Ensemble member")
    parser.add_argument("-w", "--wrf_path", default="/glade/scratch/wrfrt/realtime_ensemble/rt2020/POST/",
                        help="Path to compressed wrf output")
    parser.add_argument("-o", "--out_path", default="/glade/scratch/dgagne/rt2020/",
                        help="Path to output uncompressed files")
    args = parser.parse_args()
    dates = pd.date_range(args.start, args.end, freq="1D")
    for date in dates:
        print(date)
        date_str = date.strftime("%Y%m%d00")
        run_in_path = join(args.wrf_path, date_str, "post", args.mem)
        print(run_in_path)
        if exists(run_in_path):
            print(run_in_path + " exists")
            run_out_path = join(args.out_path, date_str, args.mem)
            if not exists(run_out_path):
                makedirs(run_out_path)
            diag_files = sorted(glob(join(run_in_path, "diags_d01_f*.nc.gz")))
            for diag_file in diag_files:
                print(diag_file)
                diag_out_file = join(run_out_path, diag_file.split("/")[-1][:-3])
                print(diag_out_file)
                subprocess.run(" ".join(["gunzip", "-c", diag_file, ">", diag_out_file]), shell=True)
    return


if __name__ == "__main__":
    main()
