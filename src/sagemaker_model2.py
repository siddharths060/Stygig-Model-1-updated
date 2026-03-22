import boto3
import json

# Replace this with your actual endpoint name
MODEL2_ENDPOINT = "YOUR_SAGEMAKER_ENDPOINT_NAME"

runtime = boto3.client("sagemaker-runtime")


def call_model2(image_bytes):
    """
    Sends image to Model-2 SageMaker endpoint
    """

    response = runtime.invoke_endpoint(
        EndpointName=MODEL2_ENDPOINT,
        ContentType="application/x-image",
        Body=image_bytes
    )

    result = json.loads(response["Body"].read())

    return result