import importlib
import subprocess
import sys

# map: tên package trong requirements -> tên module khi import
REQUIREMENTS = {
    "python-chess": "chess",
    "torch": "torch",
    "numpy": "numpy",
    "tqdm": "tqdm",
}

def check_and_install():
    missing = []
    for pkg, module in REQUIREMENTS.items():
        try:
            importlib.import_module(module)
            print(f"  [OK]      {pkg} (module '{module}') đã có")
        except ImportError:
            print(f"  [MISSING] {pkg} (module '{module}') chưa có")
            missing.append(pkg)

    if not missing:
        print("\nMôi trường đã đầy đủ, không cần cài thêm.")
        return

    print(f"\nĐang cài {len(missing)} package còn thiếu: {missing}")
    for pkg in missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

if __name__ == "__main__":
    check_and_install()