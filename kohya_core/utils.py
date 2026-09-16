# -*- coding: utf-8 -*-
"""通用工具函数（渐进式拆分，从 Kohya一键工具.py 迁出）。
原文件通过 `from kohya_core.utils import *` 使用。
"""
import os
import sys
import re
import queue
import subprocess
import shutil
import json
import threading
import time
import urllib.request
import socket
from urllib.parse import urlparse

from kohya_core.configs import PY_MIN, PY_MAX
from kohya_core.paths import get_kohya_dir

# 手动停止 / 进程管理的全局状态（迁移到本包时一并保留，供 run_stream / stop_active_process / reset_stop 使用）
_STOP_EVENT = threading.Event()
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_PROC = None

# 显式导出全部名字（含下划线开头），供 `from kohya_core.utils import *` 使用。
__all__ = [
    "StopRequested", "format_eta", "_terminate_tree", "stop_active_process", "reset_stop", "active_process_pids",
    "build_env", "build_direct_env", "clear_proxy_env", "proxy_reachable", "run_stream", "_download", "find_git", "_py_version", "find_python",
    "venv_python", "_yq", "split_triggers", "system_proxy",
]

class StopRequested(Exception):
    """用户手动停止当前任务（训练/预处理/安装等）。"""

def format_eta(sec):
    """把秒数格式化成 时:分:秒。"""
    if sec is None or sec < 0:
        return "--:--"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _terminate_tree(proc):
    """Windows 上终止整个进程树（accelerate 会拉起训练子进程）。"""
    try:
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True, timeout=30,
        )
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass

def stop_active_process():
    """请求停止当前正在运行的子进程（训练/预处理/安装等）。"""
    global _ACTIVE_PROC
    _STOP_EVENT.set()
    with _ACTIVE_LOCK:
        proc = _ACTIVE_PROC
    if proc is not None and proc.poll() is None:
        _terminate_tree(proc)
    return True

def active_process_pids():
    """返回当前活跃子进程（训练/预处理/安装等）的 PID + 全部子进程 PID（用于小工具「清理显存」排除，
    防止一边训练一边点清理把正在跑的训练杀了）。无活跃进程返回空集合。"""
    with _ACTIVE_LOCK:
        proc = _ACTIVE_PROC
    if proc is None or proc.poll() is not None:
        return set()
    pids = {proc.pid}
    try:
        code = (
            "$m=@{}; Get-CimInstance Win32_Process | ForEach-Object { $m[$_.ProcessId]=[int]$_.ParentProcessId }; "
            "$roots=@(%d); $out=New-Object System.Collections.Generic.HashSet[int]; "
            "$q=New-Object System.Collections.Generic.Queue[int]; foreach($r in $roots){ [void]$q.Enqueue($r) }; "
            "while($q.Count -gt 0){ $cur=$q.Dequeue(); foreach($k in $m.Keys){ if($m[$k] -eq $cur -and -not $out.Contains($k)){ [void]$out.Add($k); [void]$q.Enqueue($k) } } }; "
            "$out | ForEach-Object { $_ }" % proc.pid
        )
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", code],
                           capture_output=True, timeout=20)
        for ln in (r.stdout or b"").decode("utf-8", errors="replace").splitlines():
            ln = ln.strip()
            if ln.isdigit():
                pids.add(int(ln))
    except Exception:
        pass
    return pids


def reset_stop():
    """开始新任务前调用，清除上一次的停止信号。"""
    _STOP_EVENT.clear()

_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def proxy_reachable(proxy):
    """判断代理地址是否可连接；本机代理端口关闭时返回 False。"""
    if not proxy:
        return False
    try:
        raw = proxy if "://" in proxy else "http://" + proxy
        u = urlparse(raw)
        host, port = u.hostname, u.port
        if not host or not port:
            return False
        if host.lower() in ("127.0.0.1", "localhost", "::1"):
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.5)
                return sock.connect_ex((host, port)) == 0
        return True
    except Exception:
        return False


def clear_proxy_env(env):
    """清除 requests/pip/curl 代理变量，用于国内镜像直连。"""
    for key in _PROXY_ENV_KEYS:
        env.pop(key, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    return env


def build_env(extra_dirs=()):
    env = dict(os.environ)
    # PATH 必须先拆成独立目录再过滤。旧实现把完整 PATH 当成一个元素，
    # 因此永远无法命中/移除 PyInstaller 目录，外部 venv 仍会被 DLL 污染。
    paths = list(extra_dirs) + env.get("PATH", "").split(os.pathsep)
    # 清除会污染外部 Python 子进程的变量：打包版 / PATH 上其他 Python 可能
    # 通过 PYTHONHOME/PYTHONPATH 把错误的 DLL、模块塞进 venv 子进程
    # （典型现象：venv 是 Python 3.10，却报 "Module use of python312.dll conflicts"）。
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONSTARTUP", None)
    # 移除 PyInstaller 解包目录：它的 python312.dll 等 DLL 不应出现在外部
    # Python 子进程的 PATH 搜索里，避免跨 Python 版本 DLL 冲突。
    _meipass = getattr(sys, "_MEIPASS", None)
    blocked = set()
    if _meipass:
        blocked.add(os.path.normcase(os.path.abspath(_meipass)))
    if getattr(sys, "frozen", False):
        blocked.add(os.path.normcase(os.path.abspath(os.path.dirname(sys.executable))))
    paths = [p for p in paths if p and
             os.path.normcase(os.path.abspath(p.strip().strip('"'))) not in blocked]
    env["PATH"] = os.pathsep.join(paths)
    # 清除已经失效的代理，避免国内镜像被送进已关闭的 Clash/v2ray 端口。
    for key in _PROXY_ENV_KEYS:
        value = env.get(key)
        if value and not proxy_reachable(value):
            env.pop(key, None)
    # 网络不稳时 pip 容易 IncompleteRead 中断：全局加大重试次数与超时
    env.setdefault("PIP_RETRIES", "10")
    env.setdefault("PIP_TIMEOUT", "120")
    # 子进程 stdout/stderr 强制 UTF-8：英文系统（cp1252）下训练/预处理打印中文会
    # UnicodeEncodeError（accelerator.print 输出含中文项目名/trigger/caption 时崩溃）。
    # setdefault 不覆盖用户已有的 PYTHONIOENCODING；run_stream 父进程本就按 utf-8 解码。
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # 关键：子进程 stdout 被 run_stream 接管成 PIPE 后，Python 默认用「块缓冲」(4~8KB)，
    # 而不是控制台下的行缓冲。后果是 —— 进程正常退出时缓冲会 flush，日志完整；
    # 但进程**中途硬崩**（DLL 加载失败 / 段错误 / 被安全软件杀 / 进程树被清理）时
    # **缓冲区整块丢失，父进程一个字都收不到**，于是报错变成「预处理失败，请查看上方日志」
    # 而上方日志是空的（2026-09-15 qionglora 用户实测：preprocess.py 两轮零输出）。
    # 强制无缓冲后，崩溃前已打印的内容一定先落进父进程日志，失败才可诊断。
    env["PYTHONUNBUFFERED"] = "1"
    # huggingface_hub 1.x 默认走 Xet 协议（直连 cas-server.xethub.hf.co），国内常报
    # 401/超时且绕过 hf-mirror 镜像（如第三引擎下载 12GB+ 模型失败）；全局禁用，
    # 回退经典 HTTP 下载（走 HF_ENDPOINT 镜像）。
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    # Windows 不支持 PYTORCH_CUDA_ALLOC_CONF=expandable_segments（PyTorch 会提示
    # "expandable_segments not supported on this platform"）。网上流传的 Linux 优化命令
    # （setx PYTORCH_CUDA_ALLOC_CONF "expandable_segments:True"）在 Windows 无效，
    # 且会让分配器回退异常，出现「显存剩 14GB 却连 192MB 都分配失败」的假性 OOM
    # （2026-08-30 RTX 5060 Ti 用户实测，VAE 缓存 latent 阶段 100% 复现）。
    # 训练子进程剥离该配置，保留其他有效项（如 max_split_size_mb）。
    _alloc = env.get("PYTORCH_CUDA_ALLOC_CONF")
    if _alloc:
        _kept = [p.strip() for p in _alloc.split(",")
                 if p.strip() and "expandable_segments" not in p.lower()]
        if _kept:
            env["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(_kept)
        else:
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    return env


def build_direct_env(extra_dirs=()):
    """构造国内镜像直连环境：不继承系统/历史代理。"""
    return clear_proxy_env(build_env(extra_dirs))


# ---- run_stream 的收尾策略（2026-09-16）----
# ⚠️ **结束判据是「进程退出」，不是「管道 EOF」。**
# Windows 上子进程的 stdout 管道句柄会被**孙进程**继承（git clone 的 git-remote-https、
# pip 的构建子进程、杀软扫描进程等）；只要还有任何一个孙进程握着它，管道就永不 EOF。
# 旧实现用 `for line in proc.stdout` 阻塞读 → 孙进程不放手就**永久卡住**，
# 表现为「活已经干完了，但任务一直不结束」，用户只能手动结束任务。
#
# 实测（2026-09-16 qiansui 用户）：下载训练内核**每次都在最后卡住**；
# 手动结束后重点一次，工具检测完直接说「已安装」→ 证明阻塞发生在收尾而非安装本身。
# （手动结束是**不受控中断**，半装的 venv/半下的 wheel 都可能留下隐患 —— 必须修掉。）
_STREAM_POLL = 0.15        # 主循环轮询间隔（秒）
_STREAM_TAIL_GRACE = 2.0   # 进程退出后，额外排空管道缓冲的宽限时间（秒）
_STREAM_KILL_WAIT = 10.0   # 停止路径收尾时，等进程真正结束的最长时间（秒）


def run_stream(cmd, cwd=None, env=None, logf=print, collect=None):
    """运行命令并把 stdout/stderr 实时交给 logf。返回退出码。

    支持手动停止：stop_active_process() 会终止当前进程树，
    并在读取循环中抛出 StopRequested（调用方按“用户主动停止”处理）。

    ⚠️ 结束判据是 **proc.poll()（进程已退出）**，不是「管道读到 EOF」——
    孙进程可能长期持有管道句柄，等 EOF 会让任务永久卡住（见上方常量注释）。
    读取放在 daemon 线程里，即使它被孙进程卡住也不影响主流程收尾。
    """
    global _ACTIVE_PROC
    if env is None:
        # 默认用工具标准环境：强制子进程 stdout 用 UTF-8（PYTHONIOENCODING），
        # 防止英文(cp1252)/繁体(cp950)等系统打印中文路径/日志时 UnicodeEncodeError 崩溃
        # （krea2_cache_latents 等调用漏传 env 时会命中此默认值）。
        env = build_env()
    if logf:
        logf("$ " + " ".join(str(x) for x in cmd))
    if collect is not None:
        try:
            collect.append("$ " + " ".join(str(x) for x in cmd))
        except Exception:
            pass
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=cwd, env=env, text=True, encoding="utf-8", errors="replace",
        bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    with _ACTIVE_LOCK:
        _ACTIVE_PROC = proc
    if _STOP_EVENT.is_set():
        _terminate_tree(proc)

    # 读取线程只负责「把行搬进队列」——被孙进程卡住也无所谓，主线程不再等它。
    _lines = queue.Queue()
    _EOF = object()

    def _reader():
        try:
            for _raw in proc.stdout:
                _lines.put(_raw)
        except Exception:
            pass
        finally:
            _lines.put(_EOF)

    def _emit(raw):
        _l = raw.rstrip("\n").rstrip("\r")
        if logf:
            logf(_l)
        if collect is not None:
            try:
                collect.append(_l)
            except Exception:
                pass

    def _drain():
        """把队列里已到的行全部吐出；返回是否见到 EOF。"""
        while True:
            try:
                _item = _lines.get_nowait()
            except queue.Empty:
                return False
            if _item is _EOF:
                return True
            _emit(_item)

    threading.Thread(target=_reader, daemon=True).start()
    _rc = None
    try:
        while True:
            _drain()
            _rc = proc.poll()
            if _rc is not None:
                # 进程已退出 = 真正的结束。再给读取线程一小段时间把管道里缓冲的尾巴
                # 收干净，但**绝不为它无限等待**（孙进程可能一直握着管道不放）。
                _t0 = time.time()
                while time.time() - _t0 < _STREAM_TAIL_GRACE:
                    if _drain():
                        break
                    time.sleep(_STREAM_POLL)
                _drain()
                break
            if _STOP_EVENT.is_set():
                _terminate_tree(proc)
                if logf:
                    logf("[停止] 已收到停止请求，正在终止进程…")
                break
            time.sleep(_STREAM_POLL)
    finally:
        with _ACTIVE_LOCK:
            if _ACTIVE_PROC is proc:
                _ACTIVE_PROC = None
    if _rc is None:
        # 停止路径（或进程尚未退出）：带超时收尾，别在收尾环节再造出一次挂死
        try:
            proc.wait(timeout=_STREAM_KILL_WAIT)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
    # ⚠️ 这里**绝不能** `proc.stdout.close()`。
    # 读取线程此刻正阻塞在这个管道上、持着 io 内部锁，close() 会一直等它放锁 ——
    # 实测（2026-09-16，孙进程持有管道时）close() 自身阻塞了 **29.92 秒**，
    # 等于把刚修好的挂死从循环又搬到了收尾处。
    # 读取线程是 daemon：等管道真正 EOF 时它会自己读完并退出，无需我们插手。
    if _STOP_EVENT.is_set():
        raise StopRequested("任务已手动停止")
    return proc.returncode

def _download(url, dest, logf=print):
    """带进度地下载文件。"""
    logf(f"[下载] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        last = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total and got - last > 5 * 1024 * 1024:
                last = got
                logf(f"[下载] {got / 1048576:.0f}/{total / 1048576:.0f} MB")
    logf(f"[下载] 完成 -> {dest}")

def find_git():
    cands = []
    p = shutil.which("git")
    if p:
        cands.append(p)
    for c in (
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Programs\Git\cmd\git.exe"),
    ):
        if os.path.isfile(c):
            cands.append(c)
    for c in cands:
        try:
            r = subprocess.run([c, "--version"], capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                return c
        except Exception:
            pass
    return None

def _py_version(p):
    try:
        r = subprocess.run(
            [p, "-c", "import sys;print('%d.%d.%d'%sys.version_info[:3])"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0:
            s = r.stdout.strip()
            parts = tuple(int(x) for x in s.split("."))
            return s, parts
    except Exception:
        pass
    return None, None

def _py_has_venv(p):
    """校验该 python 能创建虚拟环境（有 venv/ensurepip）。

    排除 ComfyUI 等便携版自带的精简嵌入式 python（能跑、版本正常，
    但没有 venv 模块，拿去 `-m venv` 会报 No module named venv）。
    """
    try:
        r = subprocess.run([p, "-c", "import venv, ensurepip"],
                           capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def find_python():
    """找一个能创建虚拟环境的 Python，优先 3.12（与内置 cp312 wheel 匹配），
    其次 3.11 / 3.10；PATH 里的 python 放最后（可能是精简版/版本不可控）。

    之前 PATH python 排最前：若用户 PATH 里是 3.10（或 ComfyUI 精简版），
    建出来的 venv 是 3.10，而内置离线 wheel 是 cp312，导致 numpy/torch 装不上
    （训练只有 CPU 版 torch → 'accelerator device: cpu' 卡死）。"""
    cands = []
    # 标准安装路径全扫一遍（3.12 优先）：PATH 里即使只有 ComfyUI 等精简 python，
    # 也能找到真正可建 venv 的 Python；且 3.12 与内置离线 wheel（cp312）匹配。
    for c in (
        os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Programs\Python\Python312\python.exe"),
        r"C:\Python312\python.exe",
        r"C:\Program Files\Python312\python.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Programs\Python\Python311\python.exe"),
        r"C:\Python311\python.exe",
        r"C:\Program Files\Python311\python.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Programs\Python\Python310\python.exe"),
        r"C:\Python310\python.exe",
        r"C:\Program Files\Python310\python.exe",
    ):
        if os.path.isfile(c):
            cands.append(c)
    p = shutil.which("python")
    if p:
        cands.append(p)
    for c in cands:
        s, parts = _py_version(c)
        if not (s and PY_MIN <= parts < PY_MAX):
            continue
        if not _py_has_venv(c):
            continue
        return c, s
    return None, None

def venv_python(kdir=None):
    kdir = kdir or get_kohya_dir()
    return os.path.join(kdir, "venv", "Scripts", "python.exe")

def _yq(s):
    """生成合法 yaml 字符串标量（含中文/空格/转义都安全）。"""
    import json
    return json.dumps(str(s), ensure_ascii=False)

def split_triggers(s):
    """把逗号分隔的多个 trigger 拆成列表。"""
    return [t.strip() for t in (s or "").split(",") if t.strip()]

def system_proxy():
    """读取 Windows 系统代理设置，返回代理地址或 None。"""
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                           r"Software\Microsoft\Windows\CurrentVersion\Internet Settings")
        try:
            enable, _ = winreg.QueryValueEx(k, "ProxyEnable")
            server, _ = winreg.QueryValueEx(k, "ProxyServer")
        finally:
            winreg.CloseKey(k)
        if enable and server:
            proxy = server if "://" in server else "http://" + server
            return proxy if proxy_reachable(proxy) else None
    except Exception:
        pass
    return None
