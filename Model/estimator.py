from sagemaker.estimator import Estimator
import boto3
# client = boto3.client('sagemaker').role

estimator = Estimator(image_uri="model_container", entry_point="sandbox.py", role="AWSServiceRoleForAmazonSageMakerNotebooks", instance_type="local", instance_count=1)

estimator.fit()
