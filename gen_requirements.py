import os
import subprocess

REQ_FILE = "pipreqs.txt"
ALL_FILE = "envall.txt"
OUT_FILE = "requirements.txt"
SRC_DIR = "./FaultDetector"  # 你的代码目录

def run_command(cmd):
    print(f"💻 正在运行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ 命令出错:")
        print(result.stderr)
        exit(1)
    else:
        print("✅ 命令执行成功")
        print(result.stdout.strip())

def parse_pipreqs_requirements(file_path):
    """
    解析 pipreqs 输出的 requirements.txt（只有包名，或带==）
    """
    reqs = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "==" in line:
                name, version = line.split("==")
                reqs[name.lower()] = version
            else:
                reqs[line.lower()] = None
    return reqs

def parse_conda_list_export(file_path):
    """
    解析 conda list --export 的输出（格式：包=版本）
    """
    packages = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                parts = line.split("=")
                if len(parts) >= 2:
                    name = parts[0].lower()
                    version = parts[1]
                    packages[name] = version
    return packages

def generate_files():
    run_command(f"conda list --export > {ALL_FILE}")
    run_command(f"pipreqs {SRC_DIR} --force --encoding=utf-8 --savepath {REQ_FILE}")

def merge_requirements():
    if not os.path.exists(REQ_FILE) or not os.path.exists(ALL_FILE):
        print(f"❌ 缺少必要文件: {REQ_FILE} 或 {ALL_FILE}")
        return

    used = parse_pipreqs_requirements(REQ_FILE)
    all_installed = parse_conda_list_export(ALL_FILE)

    merged = {}
    for name in used:
        if name in all_installed:
            merged[name] = all_installed[name]
        else:
            print(f"⚠️ 警告：{name} 未在 Conda 环境中找到，保留原样")
            merged[name] = used[name]

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for name, version in merged.items():
            if version:
                f.write(f"{name}=={version}\n")
            else:
                f.write(f"{name}\n")

    print(f"\n✅ 合并完成，已生成: {OUT_FILE}")

if __name__ == "__main__":
    generate_files()
    merge_requirements()
