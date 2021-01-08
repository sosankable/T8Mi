"""
Run the experiment for training
"""
# import azureml
from azureml.core import ScriptRunConfig, Dataset, Workspace, Experiment, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import Model
from azureml.tensorboard import Tensorboard


def main():
    """
    Run the experiment for training
    """
    work_space = Workspace.from_config()

    # Set up the dataset for training
    datastore = work_space.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, "datasets/mnist"))

    # Set up the experiment for training
    experiment = Experiment(workspace=work_space, name="keras-lenet-train")
#     azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 2000000000
    config = ScriptRunConfig(
        source_directory=".",
        script="train_keras.py",
        compute_target="cpu-cluster",
        arguments=[
            "--data_folder",
            dataset.as_named_input("input").as_mount(),
        ],
    )

    # Set up the Tensoflow/Keras environment
    environment = Environment("keras-environment")
    environment.python.conda_dependencies = CondaDependencies.create(
        python_version="3.7.7",
        pip_packages=["azureml-defaults", "numpy", "tensorflow==2.3.1"]
    )
    config.run_config.environment = environment

    # Run the experiment for training
    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print(
        "Submitted to an Azure Machine Learning compute cluster. Click on the link below"
    )
    print("")
    print(aml_url)    
    
    tboard = Tensorboard([run])
    # If successful, start() returns a string with the URI of the instance.
    tboard.start(start_browser=True)
    run.wait_for_completion(show_output=True)
    # After your job completes, be sure to stop() the streaming otherwise it will continue to run.
    print("Press enter to stop")
    input()
    tboard.stop()

    # Register Model
    metrics = run.get_metrics()
    run.register_model(
        model_name="keras_mnist",
        tags={"data": "mnist", "model": "classification"},
        model_path="outputs/keras_lenet.h5",
        model_framework=Model.Framework.TENSORFLOW,
        model_framework_version="2.3.1",
        properties={
            "train_loss": metrics["train_loss"][-1],
            "train_accuracy": metrics["train_accuracy"][-1],
            "val_loss": metrics["val_loss"][-1],
            "val_accuracy": metrics["val_accuracy"][-1],
        },
    )


if __name__ == "__main__":
    main()
