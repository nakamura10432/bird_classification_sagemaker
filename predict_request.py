import boto3
import json

def request2api():
    # read image data
    f = open("sample_0.jpeg", "rb")
    reqbody = f.read()
    f.close()
    
    # Request
    client = boto3.client('sagemaker-runtime')
    endpoint_response = client.invoke_endpoint(
        EndpointName='pytorch-inference-2022-08-14-04-40-53-621', 
        ContentType='image/jpeg',
        Accept='application/json',
        Body=reqbody
    )
    results = endpoint_response['Body'].read()
    detections = json.loads(results)
    print(detections)

if __name__ == '__main__':
    request2api()