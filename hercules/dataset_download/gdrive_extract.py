import os
import io
import json
import pickle
import toml
import tqdm
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

CLIENT_SECRETS_FILE = "hercules/dataset_download/client_secret.json"
SCOPES = ['https://www.googleapis.com/auth/drive']

def get_google_auth_user_info():
    creds = None
    token_path = Path("hercules/dataset_download/token.pickle")
    if token_path.exists():
        print("Loading credentials from token.pickle")
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
    creds_json = creds.to_json()
    return json.loads(creds_json)

def create_service():
    creds = Credentials.from_authorized_user_info(info=get_google_auth_user_info())
    return build('drive', 'v3', credentials=creds)

def download_files(service, file_id, filename):
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fd=fh, request=request)
        done = False

        pbar = tqdm.tqdm(
            total=100,
            desc=f"Downloading {Path(filename).stem}",
            unit="%",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            leave=False
        )

        while not done:
            status, done = downloader.next_chunk()
            if status:
                pbar.update(int(status.progress() * 100) - pbar.n)

        fh.seek(0)
        with open(filename, 'wb') as f:
            f.write(fh.read())

        pbar.close()
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")

def main(output_folder, max_workers=4):
    to_download = toml.load("hercules/dataset_download/to_download.toml")
    metadata = json.load(open("hercules/dataset_download/hercules.json"))
    service = create_service()
    out_folder = Path(output_folder)

    if not out_folder.exists():
        raise Exception(f"Output folder {out_folder} does not exist. Please create it first.")

    for obj in metadata:
        current_folder = obj.get("folder")
        print(f"\n####### Current folder: {current_folder} #######")
        if current_folder in to_download.get("folders"):
            current_folder_path = out_folder / current_folder
            current_folder_path.mkdir(parents=True, exist_ok=True)

            for sub_folder, sub_folder_files in obj.get("sub_folders")[0].items():
                sub_folder_path = current_folder_path / sub_folder
                sub_folder_path.mkdir(parents=True, exist_ok=True)

                futures = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for file_obj in sub_folder_files:
                        filename, file_link = list(file_obj.items())[0]
                        if filename in to_download.get("files"):
                            file_path = sub_folder_path / filename
                            if file_path.exists():
                                continue  # Skip if already downloaded
                            file_id = file_link.split('/')[-2] if '/' in file_link else file_link
                            futures.append(executor.submit(download_files, service, file_id, file_path))

                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Thread failed: {e}")

if __name__ == "__main__":
    output_folder = os.getenv("HERCULES_DATASET")
    if output_folder is None:
        raise EnvironmentError("HERCULES_DATASET environment variable not set.")
    
    worker_count = min(8, multiprocessing.cpu_count())
    main(output_folder=output_folder, max_workers=worker_count)
