from sagemaker.estimator import Estimator

estimator = Estimator(image_name="custom-training-container",
                      role="SageMakerRole",
                      train_instance_count=1,
                      train_instance_type="local")

estimator.fit()