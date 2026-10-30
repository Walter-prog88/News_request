"""前向检验开工第一步:按冻结清单重算 sha256,确认代码、参数与参考分布在冻结之后没有改过(前缀 fwd)。

依据 research/FORWARD_TEST_PREREG.md §1。冻结清单 research/ledger/forward_freeze_manifest.json 由
scripts/freeze_build_manifest.py 在任何人看到 2026-08-18 以后的收益之前写成。本脚本只读文件字节算哈希,
**不读任何数据内容、不 import afactor**(避免被它要校验的代码影响),也不读 cache/forward。

用法:
    python3 scripts/fwd_verify_freeze.py                 # 校验,打印结果,追加一行到 logs/fwd_verify_freeze.log
    python3 scripts/fwd_verify_freeze.py --json          # 另把结果以 JSON 打到 stdout 最后一行
    python3 scripts/fwd_verify_freeze.py --manifest X    # 校验另一份清单(测试用)

退出码(非 0 一律先停下,见预注册 §1):
    0  全部一致
    1  核心文件(src/afactor 代码、参数文件、参考分布、预注册文档、金标准、复用脚本、回归测试)不一致或缺失
    2  只有辅助文件(数据下载/质检脚本、报告渲染、上游报告正文)不一致或缺失,
       或 src/afactor 下出现清单里没有、且文件名不以 fwd 开头的新 .py
    3  清单本身不可读、格式不对,清单里的文件摘要与条目重算不符(清单被手改过),
       或清单摘要不在 logs/freeze_manifest.log 的冻结记录里(清单被整体替换过)

为什么新出现的 src/afactor/fwd*.py 不算失败:前向检验允许写自己的 fwd 适配器(只放宽日期守卫、换数据源,
不改公式,并须先在研究期上通过等价性检验,预注册 §4.4);它们不会被冻结代码 import,不会改变冻结代码的行为。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "research" / "ledger" / "forward_freeze_manifest.json"
LOG = ROOT / "logs" / "fwd_verify_freeze.log"
CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def files_digest(entries: list[dict]) -> str:
    """清单摘要 = sha256(按路径排序的 "路径\\tsha256\\n" 拼接)。与 freeze_build_manifest.files_digest 同式。"""
    h = hashlib.sha256()
    for e in sorted(entries, key=lambda x: x["path"]):
        h.update(f"{e['path']}\t{e['sha256']}\n".encode("utf-8"))
    return h.hexdigest()


def _git_head(root: Path) -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None
    except Exception:            # git 不可用不影响哈希校验
        return None


def verify(manifest_path: Path, root: Path) -> dict:
    out = dict(manifest=str(manifest_path), root=str(root), checked_at=time.strftime("%Y-%m-%d %H:%M:%S %z"),
               exit_code=0, mismatched=[], missing=[], unregistered=[], n_files=0, n_ok=0)
    try:
        man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        entries = man["files"]
        assert isinstance(entries, list) and entries and all({"path", "sha256", "category"} <= set(e) for e in entries)
    except Exception as e:                                   # 清单不可读或格式不对
        out.update(exit_code=3, error=f"清单不可读或格式不对: {type(e).__name__}: {e}")
        return out
    out["frozen_at"] = man.get("frozen_at", {})
    out["files_digest_recorded"] = man.get("files_digest")
    dg = files_digest(entries)
    out["files_digest_recomputed"] = dg
    if dg != man.get("files_digest"):
        out.update(exit_code=3, error="清单摘要与条目重算不符:清单在冻结之后被手改过")
        return out
    # 冻结脚本每次冻结都向 logs/freeze_manifest.log 追加一行摘要;清单摘要不在其中 = 清单被整体替换过
    flog = root / "logs" / "freeze_manifest.log"
    if flog.exists():
        logged = set()
        for line in flog.read_text(encoding="utf-8").splitlines():
            try:
                logged.add(json.loads(line).get("files_digest"))
            except ValueError:
                continue
        out["digest_in_freeze_log"] = dg in logged
        if dg not in logged:
            out.update(exit_code=3, error="清单摘要不在 logs/freeze_manifest.log 的冻结记录里:清单被整体替换过")
            return out
    else:
        out["digest_in_freeze_log"] = None
    core_bad = aux_bad = False          # 严重程度:核心(退出码 1)> 辅助(退出码 2),不能按数字取 max
    for e in entries:
        p = root / e["path"]
        out["n_files"] += 1
        if not p.exists():
            out["missing"].append(dict(path=e["path"], category=e["category"]))
            core_bad |= e["category"] == "core"
            aux_bad |= e["category"] != "core"
            continue
        h = sha256_file(p)
        if h != e["sha256"]:
            out["mismatched"].append(dict(path=e["path"], category=e["category"], recorded=e["sha256"][:16],
                                          now=h[:16], recorded_mtime=e.get("mtime"),
                                          now_mtime=time.strftime("%Y-%m-%d %H:%M:%S",
                                                                  time.localtime(p.stat().st_mtime))))
            core_bad |= e["category"] == "core"
            aux_bad |= e["category"] != "core"
        else:
            out["n_ok"] += 1
    listed = {e["path"] for e in entries}
    for p in sorted((root / "src" / "afactor").glob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel in listed:
            continue
        is_fwd = p.name.startswith("fwd")
        out["unregistered"].append(dict(path=rel, fwd_prefixed=is_fwd))
        if not is_fwd:
            aux_bad = True
    out["exit_code"] = 1 if core_bad else (2 if aux_bad else 0)
    out["git_head_recorded"] = man.get("git", {}).get("head")
    out["git_head_now"] = _git_head(root)
    out["history"] = man.get("history", [])
    out["prereg_params"] = man.get("prereg_params", {})
    return out


def report(res: dict) -> None:
    bar = "=" * 92
    print(bar)
    print(f"前向检验冻结校验  {res['checked_at']}")
    print(f"清单: {res['manifest']}")
    if res.get("error"):
        print(f"!!! 退出码 {res['exit_code']}: {res['error']}")
        print(bar)
        return
    fa = res.get("frozen_at", {})
    print(f"冻结时间: {fa.get('local', '?')}(UTC {fa.get('utc', '?')})  清单摘要: {res['files_digest_recorded']}")
    print(f"文件: {res['n_ok']}/{res['n_files']} 一致;清单摘要见于冻结日志: {res.get('digest_in_freeze_log')}")
    for m in res["mismatched"]:
        print(f"  [不一致][{m['category']}] {m['path']}  冻结 {m['recorded']}… -> 现在 {m['now']}…"
              f"(冻结时 mtime {m['recorded_mtime']},现在 {m['now_mtime']})")
    for m in res["missing"]:
        print(f"  [缺失][{m['category']}] {m['path']}")
    for u in res["unregistered"]:
        tag = "允许:fwd 前缀的前向代码" if u["fwd_prefixed"] else "未登记的非 fwd 模块"
        print(f"  [新文件][{tag}] {u['path']}")
    if res.get("git_head_now") and res.get("git_head_recorded") and res["git_head_now"] != res["git_head_recorded"]:
        print(f"  [提示] git HEAD 已变: 冻结时 {res['git_head_recorded'][:12]} -> 现在 {res['git_head_now'][:12]}"
              "(只提示;以文件哈希为准)")
    if len(res.get("history", [])) > 1:
        print(f"  [提示] 清单被重新冻结过 {len(res['history']) - 1} 次,逐次原因:")
        for h in res["history"]:
            print(f"      {h.get('local')}  {h.get('reason')}  digest {str(h.get('files_digest'))[:16]}…")
    pp = res.get("prereg_params", {})
    if pp:
        print("预注册关键参数(只读,前向不得修改):")
        for k in ("signal_window_start", "skip_head", "rebalance", "expected_signal_days_N", "expected_eval_days_n",
                  "last_signal_day_rule", "percentile_consistent_band"):
            if k in pp:
                print(f"  {k}: {pp[k]}")
        if isinstance(pp.get("T3"), dict) and "ie_ic_frozen" in pp["T3"]:
            print(f"  T3.ie_ic_frozen: {pp['T3']['ie_ic_frozen']}")
    code = res["exit_code"]
    msg = {0: "全部一致,可以开工",
           1: "核心文件不一致或缺失 —— 停止,上报主控,不得开始前向计算",
           2: "辅助文件不一致 / 出现未登记的非 fwd 模块 —— 停止,上报主控确认与前向检验无关后才可继续,确认写进前向报告",
           3: "清单本身有问题 —— 停止"}[code]
    print(f"结果: 退出码 {code}  {msg}")
    print(bar)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--root", default=str(ROOT), help="被校验文件的根目录(测试用)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="不写 logs/fwd_verify_freeze.log(测试用)")
    a = ap.parse_args(argv)
    res = verify(Path(a.manifest), Path(a.root))
    report(res)
    if not a.no_log:
        try:
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(time=res["checked_at"], exit_code=res["exit_code"],
                                         digest=res.get("files_digest_recomputed"),
                                         n_ok=res.get("n_ok"), n_files=res.get("n_files"),
                                         mismatched=[m["path"] for m in res.get("mismatched", [])],
                                         missing=[m["path"] for m in res.get("missing", [])],
                                         unregistered=[u["path"] for u in res.get("unregistered", [])],
                                         error=res.get("error")), ensure_ascii=False) + "\n")
        except OSError:
            pass
    if a.json:
        print(json.dumps({k: v for k, v in res.items() if k != "prereg_params"}, ensure_ascii=False, default=str))
    return int(res["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
