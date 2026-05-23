# 新手完整教程：本地代码上传/更新到 GitHub 全过程（避坑版）

本文基于实际操作踩坑整理，解决 **代理报错、文件夹已存在、缓存文件报错、推送被拒、本地远程代码冲突** 等所有新手常见问题，适配 Windows 系统、PyCharm 项目，全程命令复制即用。

## 一、前期环境准备（必做，解决GitHub连接失败）

绝大多数推送失败、连接超时、443 端口报错，都是因为残留代理，先彻底清空 Git 代理：

```bash
# 清空全局代理（核心修复命令）
git config --global --unset http.proxy
git config --global --unset https.proxy
```

## 二、首次上传代码完整流程（新项目）

### 1\. 进入本地项目文件夹

打开 CMD / PowerShell，切换到你的项目根目录（替换为自己的项目路径）：

```bash
cd D:\PyCharm 2024.1.7\daima\pythonProject\build_agent
```

### 2\. 初始化本地 Git 仓库

```bash
git init
```

### 3\. 创建忽略文件（关键！避免上传垃圾/敏感文件）

Windows 新手手动创建容易带后缀，直接用命令生成 `\.gitignore` 文件：

```bash
echo. > .gitignore
```

打开生成的 `\.gitignore`，粘贴以下通用规则（适配Python项目）：

```plain
# Python缓存文件
__pycache__/
*.pyc
*.pyo
*.pyd

# 敏感环境配置（防止密钥泄露）
.env
.env.*

# IDE配置文件
.idea/
.vscode/
*.iml

# 日志、临时文件
*.log
*.tmp
```

### 4\. 关联远程 GitHub 仓库

替换为自己的仓库地址：

```bash
git remote add origin https://github.com/1934896060lxawd-sketch/agent.git
```

### 5\. 暂存、提交本地代码

```bash
# 暂存所有代码（自动忽略gitignore配置的文件）
git add .

# 提交代码（双引号内可自定义更新说明）
git commit -m "首次上传完整项目代码"
```

### 6\. 拉取远程代码合并（解决推送被拒）

首次推送必报 **fetch first 错误**，执行合并命令：

```bash
git pull origin master --allow-unrelated-histories
```

### 7\. 推送到 GitHub 远程仓库

```bash
git push origin master
```

## 三、后续更新代码流程（日常重复操作）

首次上传完成后，后续修改本地代码，只需执行最简四步，无需重复初始化、关联仓库：

```bash
# 1. 进入项目目录
cd D:\PyCharm 2024.1.7\daima\pythonProject\build_agent

# 2. 暂存所有修改
git add .

# 3. 提交修改（备注更新内容）
git commit -m "更新Agent功能、修复bug"

# 4. 拉取远程最新代码+推送
git pull origin master
git push origin master
```

## 四、全程高频报错解决方案（对应本次所有踩坑）

### 报错1：fatal: not in a git directory

原因：未加全局参数清空代理 \| 解决：执行本文第一步全局清空代理命令

### 报错2：fatal: destination path \&\#39;agent\&\#39; already exists

原因：本地已存在同名仓库文件夹 \| 解决：

1\. 无需克隆，直接进入现有文件夹更新代码；2\. 或删除旧文件夹 `rmdir /s /q agent` 后重新克隆

### 报错3：Untracked files（\.env、\_\_pycache\_\_）

原因：未配置忽略文件 \| 解决：配置 `\.gitignore`，忽略缓存和敏感文件，禁止单独提交此类文件

### 报错4：master \-\&gt; master \(fetch first\) 推送被拒

原因：远程仓库有本地没有的文件/提交记录 \| 解决：执行 `git pull origin master \-\-allow\-unrelated\-histories` 合并后再推送

### 报错5：无法连接 github\.com 443端口

原因：本地代理残留 \| 解决：重新执行清空全局代理命令，重启终端重试

## 五、状态判断说明

执行 `git status` 提示 **nothing to commit, working tree clean** = 本地代码全部提交完毕，无任何修改、无未追踪文件，可以直接推送。

> （注：文档部分内容可能由 AI 生成）
