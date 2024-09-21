import subprocess
import time
import os


class WeightsDownloader:
    @staticmethod
    def download_if_not_exists(url, dest):
        if not os.path.exists(dest):
            WeightsDownloader.download(url, dest)

    @staticmethod
    def download(url, dest):
        start = time.time()
        print(f"downloading url: {url}")
        print(f"downloading to: {dest}")

        # Check file extension to handle .tar differently
        if url.endswith('.tar'):
            # For .tar files, use -x to extract after download
            subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
        else:
            # For .safetensors or other files, download without extraction
            subprocess.check_call(["pget", url, dest], close_fds=False)

        print(f"downloading took: {time.time() - start}")
