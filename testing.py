import boto3, os
import pandas as pd
from dotenv import load_dotenv 


load_dotenv()

region_name=os.getenv("S3_STATIC_FILE_READING_REGION_NAME") 
aws_access_key_id=os.getenv("S3_STATIC_FILE_READING_ACCESS_KEY")
aws_secret_access_key=os.getenv("S3_STATIC_FILE_READING_SECRET_KEY")
bucket_name=os.getenv("S3_STATIC_FILE_READING_BUCKET_NAME")
folder_name=os.getenv("S3_STATIC_FILE_READING_FOLDER_NAME")

# print(f"region_name: {region_name}\naws_access: {aws_access_key_id}\naws_secret: {aws_secret_access_key}\nbucket_name: {bucket_name}\nfolder_name: {folder_name}")

# INFO about IAM user 
## USER: basicuser (keys are from this user) --> (this user can access only s3 files)
### BUCKET_NAME=allfilebuckets
### KEY=static_csv_data/*.csv


def read_static_csv_from_s3(
    file_name:str, 
    region_name:str, 
    aws_access_key_id:str, 
    aws_secret_access_key:str, 
    bucket_name:str, 
    folder_name:str, 
    ) -> pd.DataFrame : 

    """ 
    Read csv from s3 bucket 
    """

    ## Setting up the resources to read the csv 
    s3 = boto3.resource(
        service_name='s3',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    ## Reading the csv file from s3 
    obj = s3.Bucket(bucket_name).Object(f'{folder_name}/{file_name}').get()
    foo = pd.read_csv(obj['Body'], index_col=0)

    return foo 


def make_csv_files_global() -> None: 
  """ 
  Make the CSV files accessible globally.
  """

  global lms_df, credit_decisioning_df, location_df 

  region_name=os.getenv("S3_STATIC_FILE_READING_REGION_NAME") 
  aws_access_key_id=os.getenv("S3_STATIC_FILE_READING_ACCESS_KEY")
  aws_secret_access_key=os.getenv("S3_STATIC_FILE_READING_SECRET_KEY")
  bucket_name=os.getenv("S3_STATIC_FILE_READING_BUCKET_NAME")
  folder_name=os.getenv("S3_STATIC_FILE_READING_FOLDER_NAME")

  file_names = ["lms_data.csv", "credit-decisioning_data.csv", "location_data_200_features.csv"]

  lms_df = read_static_csv_from_s3(file_names[0], region_name, aws_access_key_id, aws_secret_access_key, bucket_name, folder_name)
  credit_decisioning_df = read_static_csv_from_s3(file_names[1], region_name, aws_access_key_id, aws_secret_access_key, bucket_name, folder_name)
  location_df = read_static_csv_from_s3(file_names[2], region_name, aws_access_key_id, aws_secret_access_key, bucket_name, folder_name)

  print("CSV made globally accessible")

make_csv_files_global() 


print(credit_decisioning_df)


# out = read_static_csv_from_s3(
#     file_name='lms_data.csv', 
#     region_name=region_name, 
#     aws_access_key_id=aws_access_key_id, 
#     aws_secret_access_key=aws_secret_access_key, 
#     bucket_name=bucket_name, 
#     folder_name=folder_name
# )





# print(out)














# s3 = boto3.resource(
#     service_name='s3',
#     region_name='ap-south-1',
#     aws_access_key_id='AKIA5GAZTN5QKLOR36PI',
#     aws_secret_access_key='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'
# )

# # for obj in s3.Bucket('allfilebuckets').objects.all():
# #     print(obj)

# obj = s3.Bucket('allfilebuckets').Object('static_csv_data/lms_data.csv').get()
# foo = pd.read_csv(obj['Body'], index_col=0)
# print(foo)