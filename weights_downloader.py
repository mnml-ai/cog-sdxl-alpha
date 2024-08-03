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
        print("downloading url: ", url)
        print("downloading to: ", dest)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        
        # Use curl to download the file
        subprocess.check_call(["curl", "-L", url, "-o", dest], close_fds=False)
        print("downloading took: ", time.time() - start)