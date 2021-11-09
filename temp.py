import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--nargs', nargs='+', default = ["111","2222"])

for _, value in parser.parse_args()._get_kwargs():
    if value is not None:
        for v in value:
            print(v)