"""工具底层实现：文件操作 + Shell 执行。Agent 通过 tools.py 的管线调用这些函数。"""

import json
import os
import subprocess
import sys
import time
from typing import Dict


# ── File Reading ────────────────────────────────────────────────────────────────

def read_file(filepath: str, max_chars: int = 80000) -> Dict:
    """Read any file from the project and return its content."""
    if not os.path.isabs(filepath):
        candidate = os.path.abspath(filepath)
        if os.path.exists(candidate):
            filepath = candidate
        else:
            filepath = os.path.abspath(os.path.join("workspace", filepath))

    if not os.path.exists(filepath):
        return {"exists": False, "path": filepath, "error": f"File not found: {filepath}", "content": ""}

    if os.path.getsize(filepath) > 10 * 1024 * 1024:
        return {"exists": True, "path": filepath, "error": "File too large (>10MB)", "content": ""}

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except Exception as e:
        return {"exists": True, "path": filepath, "error": f"Read error: {e}", "content": ""}

    full_len = len(raw)
    content = raw[:max_chars]
    structure = None
    if filepath.endswith(".json"):
        try:
            data = json.loads(raw)
            structure = {"top_keys": list(data.keys())} if isinstance(data, dict) else {"type": "list", "length": len(data)}
        except Exception:
            pass

    return {
        "exists": True, "path": filepath, "content": content,
        "char_count": full_len, "truncated": full_len > max_chars,
        "file_size_bytes": os.path.getsize(filepath),
        "extension": os.path.splitext(filepath)[1], "structure": structure,
    }


# ── File Writing ────────────────────────────────────────────────────────────────

def write_file(filepath: str, content: str, mode: str = "overwrite") -> Dict:
    """Write content to a file."""
    if not os.path.isabs(filepath):
        if filepath.startswith(("workspace/", "workspace\\", "predictors/", "predictors\\", "tests/", "tests\\")):
            filepath = os.path.abspath(filepath)
        else:
            filepath = os.path.abspath(os.path.join("workspace", filepath))

    # Block writes to core agent infrastructure
    blocked = [os.path.abspath("agent"), os.path.abspath("main.py"), os.path.abspath("utils")]
    for b in blocked:
        if filepath.startswith(b):
            return {"success": False, "error": f"Blocked: {filepath} is protected."}

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    try:
        wm = "a" if mode == "append" else "w"
        with open(filepath, wm, encoding="utf-8") as f:
            f.write(content)
        return {"success": True, "filepath": filepath, "bytes_written": len(content.encode("utf-8")), "mode": mode}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── File Listing ────────────────────────────────────────────────────────────────

def list_files(directory: str = "workspace", pattern: str = "**/*") -> Dict:
    """List files in a directory recursively."""
    import glob as _glob
    if not os.path.isabs(directory):
        directory = os.path.abspath(directory)

    if pattern in ("*", ""):
        pattern = "**/*"

    search_path = os.path.join(directory, pattern)
    files = sorted(_glob.glob(search_path, recursive=True))

    result = []
    for f in files[:80]:
        try:
            stat = os.stat(f)
            result.append({"path": f, "size_kb": round(stat.st_size / 1024, 1),
                          "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))})
        except OSError:
            result.append({"path": f, "size_kb": 0, "modified": "unknown"})

    return {"directory": directory, "pattern": pattern, "count": len(result),
            "total_found": len(files), "overflow": len(files) > 80, "files": result}


# ── Shell Execution ─────────────────────────────────────────────────────────────

# Project root — derived once at import time
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _filter_stderr(stderr: str) -> str:
    """Remove harmless MSYS2 warnings from stderr."""
    import re as _re
    return _re.sub(r'bash\.exe: warning: could not find /tmp, please create!\r?\n?', '', stderr)


def _detect_bash() -> tuple:
    """Detect Git Bash on Windows. Returns (bash_exe, True) or (None, False)."""
    if sys.platform != "win32":
        return None, False

    def _is_wsl_bash(path: str) -> bool:
        if not path:
            return True
        p = path.lower().replace("\\", "/")
        if "windows/system32/bash" in p:
            return True
        _dir = os.path.dirname(path)
        if os.path.exists(os.path.join(_dir, "git.exe")) or os.path.exists(os.path.join(_dir, "git")):
            return False
        if "git" in p.lower():
            return False
        return False

    # ── Bundled Git Bash (vendor/bash/) — highest priority ──
    bundled = os.path.abspath(os.path.join(_PROJECT_ROOT, "vendor", "bash", "bash.exe"))
    if os.path.exists(bundled):
        return bundled, True

    import shutil as _shutil
    bash_from_path = _shutil.which("bash")
    if bash_from_path and not _is_wsl_bash(bash_from_path):
        return bash_from_path, True

    for candidate in [
        r"D:\Git\bin\bash.exe", r"D:\Git\usr\bin\bash.exe",
        r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe",
        r"C:\Git\bin\bash.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\bin\bash.exe"),
        os.path.expandvars(r"%ProgramFiles%\Git\bin\bash.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Git\bin\bash.exe"),
    ]:
        if os.path.exists(candidate):
            return candidate, True
    return None, False


def run_shell(command: str, timeout: int = 3600, working_dir: str = None,
              env_vars: dict = None) -> Dict:
    """Run a shell command with real-time streaming output."""
    if working_dir is None:
        working_dir = _PROJECT_ROOT
    working_dir = os.path.abspath(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    # Block dangerous find /
    import re as _re
    if _re.search(r'\bfind\s+/(?:\s|$|-)', command.strip(), _re.IGNORECASE):
        return {"success": False, "error": "BLOCKED: find / — use find . or find \"D:/afac2026-agent\"", "stdout": "", "stderr": ""}

    bash_exe, use_bash = _detect_bash()

    # Log shell type on first run
    if not hasattr(run_shell, "_shell_logged"):
        if use_bash:
            print(f"     [tools] Shell: Git Bash ({bash_exe})", flush=True)
        else:
            print(f"     [tools] Shell: cmd.exe (Git Bash not found, may cause issues)", flush=True)
        run_shell._shell_logged = True

    # Env — ensure venv Python is on PATH, and force unbuffered stdout
    run_env = os.environ.copy()
    run_env["PYTHONUNBUFFERED"] = "1"
    # Bundled bash needs a /tmp directory
    bundled_tmp = os.path.join(_PROJECT_ROOT, "vendor", "bash", "tmp")
    if os.path.isdir(bundled_tmp):
        run_env["TMPDIR"] = bundled_tmp
        run_env["TMP"] = bundled_tmp
        run_env["TEMP"] = bundled_tmp
    run_env["PYTHONIOENCODING"] = "utf-8"
    if sys.executable:
        venv_scripts = os.path.dirname(sys.executable)
        if os.name == "nt":
            run_env["PATH"] = venv_scripts + os.pathsep + run_env.get("PATH", "")
        else:
            run_env["PATH"] = venv_scripts + ":" + run_env.get("PATH", "")
    if env_vars:
        run_env.update(env_vars)

    # Popen + streaming
    try:
        import threading
        if use_bash:
            proc = subprocess.Popen(
                [bash_exe, "-c", command],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                cwd=working_dir, env=run_env,
            )
        else:
            proc = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                cwd=working_dir, env=run_env,
            )

        stdout_lines = []
        stderr_lines = []

        def _read_stream(stream, collector):
            for line in iter(stream.readline, ""):
                stripped = line.rstrip("\n")
                if stripped:
                    if "could not find /tmp" not in stripped:
                        print(f"     {stripped}", flush=True)
                collector.append(line)
            stream.close()

        t_out = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_lines))
        t_err = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_lines))
        t_out.start(); t_err.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill(); t_out.join(timeout=2); t_err.join(timeout=2)
            return {"success": False, "error": f"Timeout after {timeout}s", "stdout": "", "stderr": ""}

        t_out.join(timeout=5); t_err.join(timeout=5)
        return {
            "success": proc.returncode == 0, "return_code": proc.returncode,
            "stdout": "".join(stdout_lines)[:80000], "stderr": _filter_stderr("".join(stderr_lines)[:10000]),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "stdout": "", "stderr": ""}


# ── Background Shell (non-blocking, for real-time monitoring) ─────────────────

import threading as _threading

# Global registry: pid → {proc, stdout_buf, stderr_buf, start_time, command, status}
_running_processes: dict = {}
_next_pid: int = 0


def _bg_read_stream(stream, buf: list, lock: _threading.Lock):
    """Read stream into shared buffer, line by line, thread-safe."""
    try:
        for line in iter(stream.readline, ""):
            with lock:
                buf.append(line)
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def start_shell(command: str, timeout: int = 3600, working_dir: str = None) -> Dict:
    """非阻塞启动 shell 命令。立即返回进程ID，Agent 可以用 check_shell 监控输出，用 kill_shell 提前终止。

    用于长时间训练脚本：先 start_shell 启动，每隔几秒 check_shell 查看输出，
    如果发现异常（CUDA error/Traceback/loss异常）立即 kill_shell 终止，修代码后重新启动。
    """
    global _next_pid
    if working_dir is None:
        working_dir = _PROJECT_ROOT
    working_dir = os.path.abspath(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    bash_exe, use_bash = _detect_bash()

    run_env = os.environ.copy()
    run_env["PYTHONUNBUFFERED"] = "1"  # Force stdout flush on every write (pipe is not TTY)
    bundled_tmp = os.path.join(_PROJECT_ROOT, "vendor", "bash", "tmp")
    if os.path.isdir(bundled_tmp):
        run_env["TMPDIR"] = bundled_tmp
        run_env["TMP"] = bundled_tmp
        run_env["TEMP"] = bundled_tmp
    if sys.executable:
        venv_scripts = os.path.dirname(sys.executable)
        if os.name == "nt":
            run_env["PATH"] = venv_scripts + os.pathsep + run_env.get("PATH", "")
        else:
            run_env["PATH"] = venv_scripts + ":" + run_env.get("PATH", "")

    pid = _next_pid
    _next_pid += 1
    stdout_buf, stderr_buf = [], []
    lock = _threading.Lock()

    try:
        if use_bash:
            proc = subprocess.Popen(
                [bash_exe, "-c", command],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                cwd=working_dir, env=run_env,
            )
        else:
            proc = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                cwd=working_dir, env=run_env,
            )

        t_out = _threading.Thread(target=_bg_read_stream, args=(proc.stdout, stdout_buf, lock), daemon=True)
        t_err = _threading.Thread(target=_bg_read_stream, args=(proc.stderr, stderr_buf, lock), daemon=True)
        t_out.start(); t_err.start()

        _running_processes[pid] = {
            "proc": proc, "stdout_buf": stdout_buf, "stderr_buf": stderr_buf,
            "lock": lock, "start_time": time.time(), "command": command,
            "timeout": timeout, "t_out": t_out, "t_err": t_err,
            "last_read_pos": 0,  # stdout bytes already returned
        }

        return {
            "success": True, "pid": pid, "status": "running",
            "command": command[:200],
            "hint": f"进程 {pid} 已启动。用 check_shell({pid}) 监控输出。用 kill_shell({pid}) 终止。",
        }
    except Exception as e:
        return {"success": False, "error": str(e), "pid": pid}


def check_shell(pid: int, wait: float = 30) -> Dict:
    """检查后台进程状态和新输出。默认阻塞等待 30 秒后返回。多进程时短超时避免僵尸进程卡住。

    用于实时监控训练脚本：检查是否有 Traceback/CUDA error/loss 异常等。
    """
    if pid not in _running_processes:
        return {"success": False, "error": f"进程 {pid} 不存在（可能已结束或被清理）", "status": "unknown"}

    info = _running_processes[pid]
    proc = info["proc"]
    lock = info["lock"]
    buf = info["stdout_buf"]
    err_buf = info["stderr_buf"]

    # Block for up to `wait` seconds, break early if:
    # - process finishes (poll returns not None)
    # - previous output exists but no new output for > 90s (zombie protection)
    last_check_pos = info["last_read_pos"]
    silent_since = time.time()
    had_output_before = info["last_read_pos"] > 0

    if wait > 0:
        deadline = time.time() + wait
        while time.time() < deadline:
            ret = proc.poll()
            if ret is not None:
                break  # Process ended, return immediately

            # Zombie protection: check if new output appeared since last check
            with lock:
                current_pos = len(buf)
            if current_pos > last_check_pos:
                last_check_pos = current_pos
                silent_since = time.time()

            # Process had output before but now silent > 25s → likely zombie
            if had_output_before and (time.time() - silent_since) > 25:
                break

            time.sleep(2)

    # Read new output
    with lock:
        new_stdout = "".join(buf[info["last_read_pos"]:])
        info["last_read_pos"] = len(buf)
        new_stderr = "".join(err_buf)

    # Check status
    ret = proc.poll()
    elapsed = time.time() - info["start_time"]
    has_ever_output = info["last_read_pos"] > 0 or bool(new_stderr)

    if ret is not None:
        status = "completed" if ret == 0 else "error"
    elif elapsed > info["timeout"]:
        try:
            proc.kill()
        except Exception:
            pass
        status = "timeout"
    elif not has_ever_output:
        status = "loading"  # Initial data loading, don't panic
    elif not new_stdout and not new_stderr:
        idle_sec = elapsed - info.get("last_output_time", 0)
        # > 45s idle + had previous output = likely zombie (Windows Git Bash quirk)
        status = "stuck" if idle_sec > 120 else "running"
    else:
        status = "running"

    if new_stdout or new_stderr:
        info["last_output_time"] = elapsed

    result = {
        "success": True, "pid": pid, "status": status,
        "elapsed": round(elapsed, 1), "new_output": new_stdout[-8000:] if new_stdout else "",
        "stderr": _filter_stderr(new_stderr[-3000:]) if new_stderr else "",
        "return_code": ret,
    }

    # Auto-detect common errors in output
    all_output = (new_stdout + new_stderr).lower()
    if any(kw in all_output for kw in ("traceback", "error:", "exception", "cuda error",
                                         "assertion", "failed.", "killed")):
        result["warning"] = "检测到错误关键词 (Traceback/Error/CUDA error 等)，建议 kill_shell 终止并修复代码"

    if status in ("completed", "error", "timeout"):
        # Clean up finished process
        info["t_out"].join(timeout=1)
        info["t_err"].join(timeout=1)
        del _running_processes[pid]

    return result


def kill_shell(pid: int) -> Dict:
    """终止后台进程及其所有子进程。用于发现训练异常时提前停止，修改代码后重新启动。"""
    if pid not in _running_processes:
        return {"success": False, "error": f"进程 {pid} 不存在", "status": "not_found"}

    info = _running_processes[pid]
    proc = info["proc"]

    with info["lock"]:
        final_stdout = "".join(info["stdout_buf"])
        final_stderr = "".join(info["stderr_buf"])

    try:
        if sys.platform == "win32":
            # Kill entire process tree (bash → python → children)
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=10,
            )
        else:
            proc.kill()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    info["t_out"].join(timeout=2)
    info["t_err"].join(timeout=2)
    del _running_processes[pid]

    return {
        "success": True, "pid": pid, "status": "killed",
        "elapsed": round(time.time() - info["start_time"], 1),
        "final_output": final_stdout[-8000:],
        "final_stderr": _filter_stderr(final_stderr[-3000:]),
    }
