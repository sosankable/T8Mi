import tensorflow as tf
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Model


environment = Environment("my-torch-environment")
environment.python.conda_dependencies = CondaDependencies.create(
    pip_packages=["azureml-defaults", "tensorflow=={}".format(tf.__version__), "numpy"]
)
