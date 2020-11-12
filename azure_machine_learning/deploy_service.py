"""
Deploy model to your service
"""

from azureml.core import Environment, Model, Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice


def main():
    """
    Deploy model to your service
    """
    work_space = Workspace.from_config()
    environment = Environment("keras-service-environment")
    environment.python.conda_dependencies = CondaDependencies.create(
        python_version="3.7.7",
        pip_packages=[
            "azureml-defaults",
            "tensorflow==2.3.1",
            "numpy",
            "gzip",
        ],
    )
    model = Model(work_space, "keras_mnist")
    service_name = "keras_mnist-service"
    inference_config = InferenceConfig(
        entry_script="score_keras.py", environment=environment
    )
    aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
    service = Model.deploy(
        workspace=work_space,
        name=service_name,
        models=[model],
        inference_config=inference_config,
        deployment_config=aci_config,
        overwrite=True,
    )
    service.wait_for_deployment(show_output=True)


if __name__ == "__main__":
    main()
