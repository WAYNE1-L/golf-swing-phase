import os
import subprocess

# 设置仓库地址
GITHUB_REPO_URL = "https://github.com/WAYNE1-L/golf-swing-phase.git"
LOCAL_REPO_DIR = "D:/GOLF"

# 切换到项目目录
os.chdir(LOCAL_REPO_DIR)

# 初始化本地 Git 仓库（如果还没初始化）
if not os.path.exists(os.path.join(LOCAL_REPO_DIR, ".git")):
    subprocess.run(["git", "init"])

# 添加远程仓库（如果没添加过）
remotes = subprocess.check_output(["git", "remote"]).decode().split()
if "origin" not in remotes:
    subprocess.run(["git", "remote", "add", "origin", GITHUB_REPO_URL])

# 添加所有文件
subprocess.run(["git", "add", "."])

# 提交更改
subprocess.run(["git", "commit", "-m", "Initial commit of golf swing phase analysis system"])

# 推送到 GitHub（主分支 main）
subprocess.run(["git", "branch", "-M", "main"])
subprocess.run(["git", "push", "-u", "origin", "main"])
