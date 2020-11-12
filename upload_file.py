import argparse
from azureml.core import Workspace


def parse_args():
    """
    Parse argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="file folder", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    work_space = Workspace.from_config()
    datastore = work_space.get_default_datastore()
    datastore.upload(src_dir="./data", target_path=args.folder, overwrite=True)


if __name__ == "__main__":
    main()
