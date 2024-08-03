import os
import requests
from tqdm import tqdm
from omegaconf import OmegaConf

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        block_size = 8192
        with open(local_filename, 'wb') as f, tqdm(
            desc=local_filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in r.iter_content(block_size):
                size = f.write(data)
                progress_bar.update(size)

def download_ip_adapter(save_path):
    os.makedirs(save_path, exist_ok=True)

    config_url = "https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/ip_adapter/config.py"
    response = requests.get(config_url)
    config = OmegaConf.create(response.text)

    sdxl_models = config.sdxl_models

    for model_name, model_info in sdxl_models.items():
        file_name = os.path.join(save_path, f"{model_name}.bin")
        if not os.path.exists(file_name):
            print(f"Downloading {model_name}...")
            download_file(model_info.path, file_name)
        else:
            print(f"{model_name} already exists. Skipping download.")

if __name__ == "__main__":
    download_ip_adapter("./ip-adapter")