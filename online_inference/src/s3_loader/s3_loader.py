from boto3.session import Session
import os

S3_URL = 'https://storage.yandexcloud.net'
S3_BUCKET = 'mlprodhw2'

MODEL_FOLDER = 'models'
MODEL_FILE = 'estimator.pkl'


def s3_loader():
    session = Session()
    s3 = session.client("s3", endpoint_url=S3_URL)
    local_output_path = os.path.join(os.getcwd(), MODEL_FOLDER, MODEL_FILE)
    print(S3_BUCKET, MODEL_FILE, local_output_path)
    s3.download_file(S3_BUCKET, MODEL_FILE, local_output_path)
