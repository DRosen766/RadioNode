from sagemaker.estimator import Estimator
import boto3
test_bucket_name = open("test_bucket_name.txt", "r").read()
s3_output_location=f's3://{test_bucket_name}/saved_model'
estimator = Estimator(image_uri="model_container", role="AWSServiceRoleForAmazonSageMakerNotebooks", output_path=s3_output_location, instance_type="local", instance_count=1)
# param defines SM_CHANNEL_TRAINING
estimator.fit({"training":f"s3://{test_bucket_name}/test/test_metadata.csv"})