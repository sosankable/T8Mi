"""
Deploy model to your service
"""
import numpy as np
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
        pip_packages=["azureml-defaults", "numpy", "tensorflow==2.3.1"],
    )
    model = Model(work_space, "keras_mnist")
    model_list = model.list(work_space)
    validation_accuracy = []
    version = []
    for i in model_list:
        validation_accuracy.append(float(i.properties["val_accuracy"]))
        version.append(i.version)
    model = Model(
        work_space, "keras_mnist", version=version[np.argmax(validation_accuracy)]
    )
    service_name = "keras-mnist-service"
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
    print(service.get_logs())


if __name__ == "__main__":
    main()
