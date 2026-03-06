---
name: deploy
description: 将本地代码 commit 并推送到 GitHub，然后 SSH 到远程服务器执行 git pull 同步代码。开发完成后用此命令部署。
argument-hint: [commit message]
allowed-tools: Bash(git add*), Bash(git commit*), Bash(git push*), Bash(git status*), Bash(git diff*), Bash(git log*), Bash(ssh*)
---

将当前本地代码提交并同步到远程服务器 g01n23_maben。

## 步骤

1. 运行 `git status` 查看当前改动。如果没有任何改动，告知用户无需提交，跳过后续步骤。

2. 运行 `git diff --stat HEAD` 展示变更摘要。

3. 运行 `git add -A` 暂存所有改动。

4. 确定 commit message：
   - 如果用户提供了 `$ARGUMENTS`，直接使用它作为 commit message
   - 如果 `$ARGUMENTS` 为空，根据 `git diff --stat HEAD` 的内容自动生成一条简洁的中文 commit message

5. 运行 `git commit -m "<commit message>"`

6. 运行 `git push origin main`
   - 如果失败，停止并告知用户错误原因，不继续执行下一步

7. 运行 `ssh g01n23_maben "cd /work/home/maben/project/epitope_prediction/GraphBepi && git pull origin main"`

8. 报告所有步骤的执行结果。

## 注意

- 如果 git push 失败，不要尝试 SSH 到远程，先解决推送问题
- 远程 git pull 失败时，完整显示错误信息
