import os
import io
import json
import pickle
import toml
import tqdm
from pathlib import Path
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

CLIENT_SECRETS_FILE = "./hercules.json"
SCOPES = ['https://www.googleapis.com/auth/drive']
local_path = os.path.join(os.getcwd(), "exports")

if not os.path.exists(local_path):
    os.makedirs(local_path)

def get_google_auth_user_info():
    creds = None
    if os.path.exists('./token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    creds_json = creds.to_json()
    return json.loads(creds_json)

def create_service():
    creds = Credentials.from_authorized_user_info(info=get_google_auth_user_info())
    return build('drive', 'v3', credentials=creds)

def download_files(service, file_id, filename):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fd=fh, request=request)
    done = False

    pbar = tqdm.tqdm(total=100, desc=f"Downloading {Path(filename).stem}", unit="%", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    while not done:
        status, done = downloader.next_chunk()
        pbar.update(int(status.progress() * 100))
    fh.seek(0)
    
    with open(filename, 'wb') as f:
        f.write(fh.read())
        f.close()

def main():
    to_download = toml.load("./to_download.toml")
    service = create_service()
    out_folder = Path(to_download.get("output_folder"))
    if not os.path.exists(out_folder):
        raise Exception(f"Output folder {out_folder} does not exist. Please create it first.")
    
    metadata = json.load(open("hercules.json"))
    for obj in metadata:
        current_folder = obj.get("folder")
        print(f"\n####### Current folder: {current_folder} #######")
        if current_folder in to_download.get("folders"):
            current_folder = out_folder / current_folder
            if not os.path.exists(current_folder):
                os.makedirs(current_folder)

            for sub_folder, sub_folder_files in obj.get("sub_folders")[0].items():
                sub_folder = current_folder / sub_folder
                if not os.path.exists(sub_folder):
                    os.makedirs(sub_folder)

                for file_obj in sub_folder_files:
                    filename, file_link = list(file_obj.items())[0]
                    if filename in to_download.get("files"):
                        file_path = os.path.join(sub_folder, filename)
                        file_id = file_link.split('/')
                        file_id = file_id[-2] if len(file_id) > 2 else file_id[-1]
                        download_files(service, file_id, file_path)

if __name__ == "__main__":
    main()
