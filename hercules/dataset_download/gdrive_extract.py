import io
import os
import re
import json
import pickle
from pathlib import Path
from typing import Iterable, List, Optional

import tqdm
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# ==== Credentials/Scopes (kept as in your original script) ====
CLIENT_SECRETS_FILE = "hercules/dataset_download/client_secret.json"
TOKEN_PICKLE_FILE   = "hercules/dataset_download/token.pickle"
SCOPES = ['https://www.googleapis.com/auth/drive']

# ==== URL/ID parsing ====
FOLDER_ID_RE = re.compile(
    r"(?:https?://)?(?:www\.)?drive\.google\.com/(?:drive/(?:u/\d+/)?folders/|open\?id=)([a-zA-Z0-9_-]+)"
)

def extract_id(url_or_id: str) -> str:
    """
    Accepts a Drive folder URL or a raw ID, returns the folder ID.
    """
    url_or_id = url_or_id.strip()
    m = FOLDER_ID_RE.match(url_or_id)
    if m:
        return m.group(1)
    # looks like a raw id
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", url_or_id):
        return url_or_id
    raise ValueError(f"Could not parse Drive folder ID from: {url_or_id}")

# ==== Auth/service ====
def get_google_auth_user_info() -> dict:
    """
    Loads/refreshes OAuth creds (compatible with your existing files).
    """
    creds = None
    if os.path.exists(TOKEN_PICKLE_FILE):
        print("Loading credentials from token.pickle")
        with open(TOKEN_PICKLE_FILE, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            print("Running local OAuth flow to obtain new credentials...")
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PICKLE_FILE, 'wb') as token:
            pickle.dump(creds, token)

    return json.loads(creds.to_json())

def create_service():
    creds = Credentials.from_authorized_user_info(info=get_google_auth_user_info())
    return build('drive', 'v3', credentials=creds)

# ==== Drive helpers ====
def get_file_metadata(service, file_id: str, fields: str = "id,name,mimeType,size,md5Checksum"):
    return service.files().get(
        fileId=file_id,
        fields=fields,
        supportsAllDrives=True
    ).execute()

def list_children(service, folder_id: str) -> Iterable[dict]:
    """
    Yields all immediate children of a folder (files and subfolders).
    """
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name,mimeType,size,md5Checksum)",
            pageToken=page_token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        for item in resp.get("files", []):
            yield item
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

def is_folder(item: dict) -> bool:
    return item.get("mimeType") == "application/vnd.google-apps.folder"

# ==== Download ====
def file_exists_with_same_size(path: Path, size_str: Optional[str]) -> bool:
    if not path.exists() or size_str is None:
        return False
    try:
        return path.stat().st_size == int(size_str)
    except Exception:
        return False

def download_file(service, file_id: str, out_path: Path, overwrite: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = get_file_metadata(service, file_id, fields="id,name,mimeType,size,md5Checksum")
    mime = meta.get("mimeType")

    # Skip Google Docs/Sheets/etc. (export mapping can be added if needed)
    if mime and mime.startswith("application/vnd.google-apps.") and mime != "application/vnd.google-apps.folder":
        print(f"Skipping Google-native doc (not a binary): {meta.get('name')} [{mime}]")
        return

    if not overwrite and file_exists_with_same_size(out_path, meta.get("size")):
        print(f"Exists (same size), skipping: {out_path}")
        return

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)

    # Write to temp .part file first for safety
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with open(tmp_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fd=fh, request=request, chunksize=10 * 1024 * 1024)
        done = False
        pbar = tqdm.tqdm(
            total=100,
            desc=f"Downloading {meta.get('name')}",
            unit="%",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        try:
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    pbar.update(int(status.progress() * 100) - pbar.n)
        finally:
            pbar.close()

    tmp_path.replace(out_path)
    print(f"Saved: {out_path}")

def download_folder_tree(service, folder_id: str, root_out: Path, overwrite: bool = False) -> None:
    # Create a subdir named as the folder itself
    base_out = root_out 
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"\n==== Folder: {base_out.name} ({folder_id}) -> {base_out} ====")

    # DFS traversal
    stack: List[tuple[str, Path]] = [(folder_id, base_out)]
    while stack:
        current_id, current_out = stack.pop()

        # List children of current folder
        for item in list_children(service, current_id):
            name = item["name"]
            if is_folder(item):
                sub_out = current_out / name
                sub_out.mkdir(parents=True, exist_ok=True)
                stack.append((item["id"], sub_out))
            else:
                out_path = current_out / name
                if out_path.parent.name in ["Radar", "Calibration", "Image"]:  #"PR_GT", "sensor_data"
                    continue
                if name.startswith("LiDAR"):
                    #out_path = current_out / "Aeva_data" / name
                    continue
                download_file(service, item["id"], out_path, overwrite=overwrite)

# ==== I/O for links ====
def read_links(path: str) -> List[str]:
    links: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            links.append(s)
    return links

# ==== Main ====
def main():

    # Resolve output dir
    out_root = os.getenv("TMP_HERCULES_DATASET")
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    # Gather link. Place links recieved from hercules via mail here.
    links= {
        "mountain_01_Day": "",
        "mountain_02_Night": "",
        "mountain_03_Day": "",

        "library_01_Day": "",
        "library_02_Night": "",
        "library_03_Day": "",

        "sports_complex_01_Day": "",
        "sports_complex_02_Night": "",
        "sports_complex_03_Day": "",

        "parking_lot_01_Day": "",
        "parking_lot_02_Day": "",
        "parking_lot_03_Night": "",
        "parking_lot_04_Day": "",

        "river_island_01_Day": "",
        "river_island_02_Day": "",
        "river_island_03_Dusk": "",

        "bridge_01_Day": "",
        "bridge_02_Night": "",

        "street_01_Day": "",

        "stream_01_Day": "",
        "stream_02_Night": ""
    }


    service = create_service()

    # Process each folder
    for folder_name, url in links.items():
        folder_id = extract_id(url)
        try:
            download_folder_tree(service, folder_id, out_root_path / folder_name, overwrite=True)
        except Exception as e:
            print(f"Error downloading {url}: {e}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
