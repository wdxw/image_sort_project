"""用 requests + 代理下载 CIFAR-10 到 torchvision 期望的位置"""
import requests, os, tarfile, time, sys

PROXY = "http://127.0.0.1:7897"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(SCRIPT_DIR, "data", "cifar-10-python.tar.gz")

os.makedirs(os.path.dirname(DEST), exist_ok=True)

# 如果文件已存在且完整，跳过
if os.path.exists(DEST):
    try:
        with tarfile.open(DEST, 'r:gz') as tar:
            tar.getmembers()
        print(f"[跳过] 文件已存在且完整 ({os.path.getsize(DEST)/1024/1024:.1f} MB)")
        sys.exit(0)
    except Exception:
        print("[提示] 已存在文件损坏，重新下载...")
        os.remove(DEST)

print(f"正在通过代理 {PROXY} 下载 CIFAR-10 (170MB)...")
proxies = {"http": PROXY, "https": PROXY}
start = time.time()

resp = requests.get(URL, proxies=proxies, stream=True, timeout=(30, 900))
resp.raise_for_status()
total = int(resp.headers.get("content-length", 0))
downloaded = 0

with open(DEST, "wb") as f:
    for chunk in resp.iter_content(chunk_size=1024*1024):
        f.write(chunk)
        downloaded += len(chunk)
        elapsed = time.time() - start
        speed = downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
        pct = downloaded * 100 / total if total > 0 else 0
        eta = (total - downloaded) / (speed * 1024 * 1024) if speed > 0 else 0
        if total > 0:
            sys.stdout.write(f"\r  {downloaded/1024/1024:.0f}/{total/1024/1024:.0f} MB "
                             f"({pct:.0f}%) {speed:.1f} MB/s ETA:{eta:.0f}s ")
            sys.stdout.flush()

elapsed = time.time() - start
size = os.path.getsize(DEST)
print(f"\n下载完成! {size/1024/1024:.1f} MB, 用时 {elapsed:.0f}s "
      f"({size/elapsed/1024/1024:.1f} MB/s)")

# 验证
with tarfile.open(DEST, 'r:gz') as tar:
    names = tar.getnames()
print(f"有效 tar, {len(names)} 个文件. 可运行 main.py 了!")
