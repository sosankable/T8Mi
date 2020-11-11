from azureml.core import Workspace
import json


def main():
    config = json.load(open("config/azureml.json", "r"))
    work_space = Workspace.create(
        name="mltibame",  # provide a name for your workspace
        subscription_id=config["subscription_id"],  # provide your subscription ID
        resource_group="Tibame",  # provide a resource group name
        create_resource_group=True,
        location="eastus2",
    )  # For example: 'westeurope' or 'eastus2' or 'westus2' or 'southeastasia'.

    # write out the workspace details to a configuration file: .azureml/config.json
    work_space.write_config(path=".azureml")


if __name__ == "__main__":
    main()
