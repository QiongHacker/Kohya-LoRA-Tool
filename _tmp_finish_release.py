# -*- coding: utf-8 -*-
"""临时脚本（用完即删）：补完 v0.16.13 的 git 推送与 GitHub Release。

为什么需要：release.py 跑到「魔搭上传」之后，输出洪峰（tqdm 每帧一行）把进程截断了，
git_push / github_release 两步从未执行 —— 远端没有 v0.16.13 标签、工作区还挂着改动。
这里**复用 release.py 自己的函数**（不重写逻辑，避免两份实现跑偏）。
"""
import importlib.util as u
import sys

sp = u.spec_from_file_location("rel", "release.py")
rel = u.module_from_spec(sp)
sp.loader.exec_module(rel)

SECRETS = rel.load_secrets()
VER = "0.16.13"
WHICH = sys.argv[1] if len(sys.argv) > 1 else "push"

# 令牌脱敏：git_push 推 Gitee 时会把带 token 的 URL 打进日志
_real_log = rel.log


def _redact(msg):
    out = str(msg)
    for _k in ("gitee_token", "github_token", "ms_token"):
        _t = SECRETS.get(_k) or ""
        if _t and len(_t) > 6:
            out = out.replace(_t, "***")
    _real_log(out)


rel.log = _redact

if WHICH == "push":
    rel.git_push(SECRETS)
elif WHICH == "gh":
    rel.github_release(VER, SECRETS)
    rel.upload_github_asset(VER, SECRETS)
else:
    print("用法: push | gh")
