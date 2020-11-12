"""
Upload data to Azure machine learning
"""
import argparse
from azureml.core import Workspace, Dataset


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="file folder", type=str)
    parser.add_argument(
        "-t", "--target_path", help="file folder in datastore", type=str
    )
    parser.add_argument("-n", "--dataname", help="name of dataset", type=str)
    args = parser.parse_args()
    return args


def main():
    """
    Upload data to Azure machine learning
    """
    args = parse_args()
    work_space = Workspace.from_config()
    datastore = work_space.get_default_datastore()
    datastore.upload(src_dir=args.folder, target_path=args.target_path, overwrite=True)
    dataset = Dataset.File.from_files(path=(datastore, args.target_path))
    dataset.register(work_space, name=args.dataname)


if __name__ == "__main__":
    main()
