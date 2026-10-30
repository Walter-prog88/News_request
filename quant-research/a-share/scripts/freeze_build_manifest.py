"""前向检验冻结清单:对代码、参数文件、参考分布与预注册文档逐个计算 sha256 并落盘(前缀 freeze,2026-10-10)。

## 为什么
前向检验(research/FORWARD_TEST_PREREG.md)要证明"看到前向收益之后没有改过任何东西"。项目的 src/afactor 里大部分
新模块(minute5 / optimizer / barra / impact / index_proxy ...)还没进 git(git status 里是 ??),git HEAD 证明不了它们的状态,
所以逐文件记 sha256。前向检验代理开工第一步跑 scripts/fwd_verify_freeze.py 与本清单比对,不一致即停。

## 清单内容(research/ledger/forward_freeze_manifest.json)
  frozen_at      冻结时间,取自 `date` 命令(本地与 UTC 各一)
  git            HEAD、分支、`git status --porcelain` 摘要(按状态码计数、全文 sha256、src/scripts/tests 三处的逐行列表)
  environment    python 与关键包版本(只记录,不校验)
  files          [{path, sha256, size, mtime, category, git}],category ∈ {core, aux}:
                   core = 前向检验会直接复用或据以判定的东西(src/afactor 全部 .py、参数文件、参考分布、预注册文档、
                          金标准、复用脚本、回归测试)—— 不一致 = 退出码 1;
                   aux  = 数据下载 / 质检脚本、报告渲染脚本、上游报告正文 —— 不一致 = 退出码 2
  files_digest   sha256(按路径排序的 "路径\\t哈希\\n"),防止有人手改清单条目
  provenance     上游中间产物的哈希(cache/_tmp,只作溯源,**不**参与校验:它们可能被其他研究正常重建)
  prereg_params  预注册的关键参数(机器可读副本;以预注册文档为准)
  history        每次冻结 / 重新冻结的时间、原因、摘要
另写 research/ledger/forward_freeze_manifest.sha256(`shasum -a 256 -c` 可直接用的格式),
并向 logs/freeze_manifest.log 追加一行。

## 规则
- 只哈希文件字节,不读任何数据内容;不碰 cache/forward、cache/sealed、data_8:21。
- 清单已存在时拒绝覆盖;确需重新冻结(例如主控确认某个辅助文件的改动与前向无关)必须
  `--force --reason "..."`,旧清单改名存档,新清单的 history 里保留全部历史。重新冻结发生在看过前向收益之后时,
  前向报告必须如实写明(预注册 §1)。

用法:
    python3 scripts/freeze_build_manifest.py                  # 首次冻结
    python3 scripts/freeze_build_manifest.py --force --reason "..."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "ledger" / "forward_freeze_manifest.json"
OUT_SHA = ROOT / "research" / "ledger" / "forward_freeze_manifest.sha256"
LOG = ROOT / "logs" / "freeze_manifest.log"
CHUNK = 1 << 20

# ---------------------------------------------------------------- 文件清单
CORE = [
    "glob:src/afactor/*.py",
    # 预注册文档与参数
    "research/FORWARD_TEST_PREREG.md",
    "research/ledger/comp_v5_members.json",          # COMP_v5 成员与符号(实际 24 个成员,见 MINUTE5_BRIDGE §3)
    "research/ledger/pead_frozen_inputs.json",       # PEAD 冻结输入(种子 20261010 等)
    "tests/golden_engine.json",                      # 引擎金标准
    "stock_sw_industry.csv",                         # 申万一级行业快照(风险模型、PEAD 匹配都用)
    # 参考分布(前向只查不重算)
    "research/ledger/m5_bridge_daily.parquet",
    "research/ledger/m5_window_excess_quantiles.csv",
    "research/ledger/m5_window_excess_dist.csv",
    "research/ledger/m5_window_diff.csv",
    "research/ledger/m5_coldstart_summary.csv",
    "research/ledger/m5_candidate_selection.csv",
    "research/ledger/m5_backtest.csv",
    "research/ledger/freeze_ref_daily.parquet",
    "research/ledger/freeze_ref_te_daily.parquet",
    "research/ledger/freeze_ref_window_dist.parquet",
    "research/ledger/freeze_ref_quantiles.csv",
    "research/ledger/freeze_ref_checks.csv",
    "research/ledger/ie_variants.csv",
    # 前向会复用的代码路径
    "scripts/m5_bridge.py",                          # COMP_v5 打分(_comp_score / load_members)、PLANS
    "scripts/m5_select_candidates.py",               # 5 分钟执行价近似的选定(测量工具)
    "scripts/m5_baostock_check.py",
    "scripts/run_comp_v5.py",                        # COMP_v5 的来历
    "scripts/run_index_enhance.py",                  # 指增 V1(JOBS、ic_dec、run_job、load_variant、te_pred)
    "scripts/run_risk_model.py",
    "scripts/run_index_proxy.py",
    "scripts/pead_build_batches.py",
    "scripts/pead_run_arms.py",
    "scripts/pead_event_level.py",
    "scripts/pead_analyze.py",
    "scripts/pead_engine_gate.py",
    "scripts/pead_data_acceptance.py",
    "scripts/fixa_benchmark_fix.py",
    "scripts/eng_make_golden.py",
    "scripts/freeze_reference_dists.py",
    "scripts/freeze_build_manifest.py",
    "scripts/fwd_verify_freeze.py",
    # 回归测试(前向开工第二步要跑)
    "tests/test_engine_regression.py",
    "tests/test_minute5.py",
    "tests/test_fixa_benchmarks.py",
    "tests/test_data_cutoff.py",
    "tests/test_optimizer.py",
    "tests/test_optimizer_pead.py",
    "tests/test_pead_engine.py",
    "tests/test_index_proxy.py",
    "tests/test_barra_risk.py",
]
AUX = [
    # 数据下载 / 质检(前向数据代理的文件;改动要上报,但不改变信号与参数)
    "scripts/fwd_data_fetch_ak.py",
    "scripts/fwd_data_fetch_bs.py",
    "scripts/fwd_data_fetch_sina_bj.py",
    "scripts/fwd_data_qa.py",
    "tests/test_fwd_data.py",
    # 上一轮的审计 / 报告渲染脚本
    "scripts/fixb_cutoff_impact.py",
    "scripts/fixb_review2_checks.py",
    "scripts/fixb_rolling_origin.py",
    "scripts/cap_capacity_curve.py",
    "scripts/cap_render_report.py",
    "scripts/cap_report_template.txt",
    "scripts/eng_cost_model_tables.py",
    "scripts/eng_review_fixes.py",
    "scripts/eng_review_edge_probe.py",
    "scripts/run_stress_test.py",
    "scripts/stress_analyze.py",
    "scripts/run_index_enhance_pead.py",
    "scripts/run_barra_attribution.py",
    "scripts/pead_plot_coverage.py",
    "tests/test_cap_capacity.py",
    "tests/test_stress_checks.py",
    # 预注册引用的上游报告正文(冻结规则的出处;后续追加审查记录会改哈希,所以只作辅助)
    "research/MINUTE5_BRIDGE.md",
    "research/INDEX_ENHANCEMENT_CSI1000.md",
    "research/EARNINGS_EVENT_RESEARCH_PROTOCOL.md",
    "research/PEAD_PREREG_ADDENDUM.md",
    "research/PEAD_REPORT.md",
    "research/ENGINE_IMPACT_MODEL.md",
    "research/RISK_MODEL.md",
    "research/CSI1000_PROXY.md",
    "research/DATA_CUTOFF_DISCIPLINE.md",
    "research/INDEX_ENHANCEMENT_PEAD.md",
    "research/CAPACITY_REPORT.md",
    "research/STRESS_TEST.md",
]
PROVENANCE = [
    "cache/_tmp/comp_v5_score_open10.parquet",      # 冻结的研究期 COMP_v5 打分(fixb:md5 d011468f)
    "cache/_tmp/m5_score_5m.parquet",                # 5 分钟口径研究期打分(桥接)
    "cache/_tmp/m5_score_1m.parquet",
    "cache/_tmp/ie_ic_comp.parquet",                 # T3 冻结 IC 的来源(2025-12-31 行 ic_dec)
    "cache/_tmp/pead_event_level_slots.parquet",     # T4 参考分布的输入
    "cache/_tmp/pead_daily_base_1m.parquet",
    "cache/_tmp/pead_daily_base_10m.parquet",
    "glob:cache/_tmp/ie_V1_ph[0-4]_daily.parquet",   # T3 / T5 参考分布的输入
    "glob:cache/_tmp/ie_V1_ph[0-4]_holdings.parquet",
    "cache/_tmp/ie_CHK_pool_ew_daily.parquet",
    "cache/_tmp/ie_CHK_pool_ew_holdings.parquet",
    "cache/_tmp/proxy_csi1000_weights.parquet",
]

# ---------------------------------------------------------------- 预注册关键参数(机器可读副本,以预注册文档为准)
PREREG_PARAMS = {
    "prereg_doc": "research/FORWARD_TEST_PREREG.md",
    "signal_window_start": "2026-09-01",
    "position_warmup": "冷启动:date_mask 起点 = 首个前向信号日 2026-09-01,此前不持仓、不计算任何策略收益",
    "skip_head": 4,
    "rebalance": 5,
    "phases": [0, 1, 2, 3, 4],
    "exec_mode": "open10(信号日 t 收盘出信号,t+1 09:31-09:40 TWAP 成交,收益在 t+2 同一窗口实现,记在 t)",
    "data_end": "2026-10-09",
    "last_signal_day_rule": "t+2 个交易日 <= 2026-10-09 的最后一个信号日",
    "expected_signal_days_N": 21,
    "expected_eval_days_n": 17,
    "calendar_alternatives": {"10-08 休市": {"N": 20, "n": 16}, "09-25 开市": {"N": 22, "n": 18}},
    "feature_only_span": ["2026-01-01", "2026-08-30"],
    "window_excess_stat": "Π(1+r_策略) − Π(1+r_基准),minute5.forward_window_excess(df, skip_head=4, how='compound')",
    "nw_t": "mean(d) / pead_engine.nw_se(d, L=5),标注'样本太短,不作推断'",
    "percentile_consistent_band": [5, 95],
    "T1": dict(desc="COMP_v5 标准排雷", version="V1000", capital=1e7, scheme="exclude", buffer=[0.20, 0.35],
               rebalance=5, costs=["old(1bp+印花税)", "CostModel(impact_y=1.0, spread_method='tick')"],
               primary_bench="同频池等权 strategy.benchmark(rebalance=5)",
               reference_keys=["T1_excl|old|pool5|n{n}", "T1_excl|Y1|pool5|n{n}"]),
    "T2": dict(desc="COMP_v5 Top20 周频(对照)", version="V1000", capital=1e7, scheme="top", n=20, rebalance=5,
               costs=["old", "Y1"], primary_bench="同频池等权",
               reference_keys=["T2_top20|old|pool5|n{n}", "T2_top20|Y1|pool5|n{n}"]),
    "T3": dict(desc="中证1000 指增 V1", optconfig="optimizer.OptConfig() 默认值", capital=1e7,
               cost="CostModel(impact_y=1.0)", phases=5, ie_ic_frozen=0.11196633374948033,
               ie_ic_frozen_source="cache/_tmp/ie_ic_comp.parquet 行 2025-12-31 列 ic_dec(s ∈ [2024-12-13, 2025-12-23] 共 250 个 IC)",
               primary_bench="中证1000 实际成分成交时点权重(取不到则 PIT 代理)",
               reference_keys=["T3_V1|Y1|proxy_exec|n{n}"]),
    "T4": dict(desc="PEAD E 臂 vs B 臂", capital=[1e6, 1e7], fees="基准费率(每边 1bp、最低 5 元、印花税 5bp)",
               primary_stat="大批次(>=50 个有效 E 槽位)D_T 的等权均值 D̄(h5,B 对照)",
               min_events=dict(event_layer="至少 1 个大批次(>=50 个有效 E 槽位)",
                               account_layer="E 槽位合计 >= 100 且 >= 3 个相位有已建仓批次"),
               reference_keys=["T4_E|base|B|n{N}|Dbar_big", "T4_E|base|B|n{N}|k_big"]),
    "T5": dict(desc="风险模型前向校验", stat="b_d = std(z_t), z_t = 主动日收益 / (事前年化 TE_t / sqrt(242))",
               reference_keys=["T5_V1|Y1|proxy_exec|n{n}|b_d", "T5_poolew|Y1|proxy_exec|n{n}|b_d"]),
    "engine_check_tolerance": "全池等权 vs 同频池:评估窗内 Σ(超额+成本) 落在 freeze_ref_quantiles.csv CHK_poolew 行 [q005, q995]",
}


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
    h = hashlib.sha256()
    for e in sorted(entries, key=lambda x: x["path"]):
        h.update(f"{e['path']}\t{e['sha256']}\n".encode("utf-8"))
    return h.hexdigest()


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120, check=True).stdout


def _expand(specs: list[str]) -> list[Path]:
    out = []
    for s in specs:
        if s.startswith("glob:"):
            hits = sorted(ROOT.glob(s[5:]))
            if not hits:
                raise FileNotFoundError(f"没有匹配 {s}")
            out += hits
        else:
            p = ROOT / s
            if not p.exists():
                raise FileNotFoundError(f"缺文件 {s}")
            out.append(p)
    for p in out:
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(("cache/forward", "cache/sealed", "data_8:21")):
            raise PermissionError(f"冻结清单不得触碰 {rel}")
    return out


def _git_file_status(porcelain: str) -> dict:
    st = {}
    for line in porcelain.splitlines():
        if len(line) > 3:
            st[line[3:].strip().strip('"')] = line[:2]
    return st


def _entry(p: Path, cat: str, gst: dict, tracked: set) -> dict:
    rel = p.relative_to(ROOT).as_posix()
    if rel in gst:
        g = {"??": "untracked", " M": "modified", "M ": "modified(staged)", "MM": "modified"}.get(gst[rel], gst[rel])
    else:
        parent_untracked = any(rel.startswith(k) and k.endswith("/") for k, v in gst.items() if v == "??")
        g = "untracked" if parent_untracked else ("tracked-clean" if rel in tracked else "untracked-or-ignored")
    s = p.stat()
    return dict(path=rel, category=cat, sha256=sha256_file(p), size=s.st_size,
                mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.st_mtime)), git=g)


def _versions() -> dict:
    from importlib import metadata
    v = dict(python=platform.python_version(), platform=platform.platform())
    for pkg in ("numpy", "pandas", "scipy", "cvxpy", "pyarrow", "baostock", "akshare"):
        try:
            v[pkg] = metadata.version(pkg)
        except Exception:
            v[pkg] = None
    return v


def build(force: bool, reason: str | None) -> dict:
    if OUT.exists() and not force:
        raise SystemExit(f"{OUT} 已存在:拒绝覆盖。确需重新冻结请用 --force --reason \"...\"(会存档旧清单并记入 history)")
    if force and not (reason and len(reason.strip()) >= 4):
        raise SystemExit("--force 必须写明 --reason")
    local = _run(["date", "+%Y-%m-%d %H:%M:%S %Z (%z)"]).strip()
    utc = _run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]).strip()
    head = _run(["git", "rev-parse", "HEAD"]).strip()
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    porcelain = _run(["git", "status", "--porcelain"])
    tracked = set(_run(["git", "ls-files"]).splitlines())
    gst = _git_file_status(porcelain)
    counts = {}
    for line in porcelain.splitlines():
        counts[line[:2]] = counts.get(line[:2], 0) + 1
    focus = [l for l in porcelain.splitlines() if l[3:].startswith(("src/", "scripts/", "tests/"))]

    core = _expand(CORE)
    aux = _expand(AUX)
    dup = {p for p in core} & {p for p in aux}
    assert not dup, f"core 与 aux 重复: {dup}"
    files = [_entry(p, "core", gst, tracked) for p in core] + [_entry(p, "aux", gst, tracked) for p in aux]
    prov = []
    for s in PROVENANCE:
        try:
            for p in _expand([s]):
                st = p.stat()
                prov.append(dict(path=p.relative_to(ROOT).as_posix(), sha256=sha256_file(p), size=st.st_size,
                                 mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))))
        except FileNotFoundError as e:
            prov.append(dict(path=s, missing=str(e)))
    digest = files_digest(files)
    hist = []
    if OUT.exists():
        old = json.loads(OUT.read_text(encoding="utf-8"))
        hist = old.get("history", [])
        arch = OUT.with_name(f"forward_freeze_manifest.{time.strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(OUT, arch)
        print(f"旧清单存档 -> {arch.name}")
    hist.append(dict(local=local, utc=utc, reason=reason or "首次冻结(前向检验预注册,看到任何 2026-08-18 以后收益之前)",
                     files_digest=digest, n_files=len(files)))
    man = dict(
        schema=1,
        purpose="前向检验(2026-09 窗口)冻结清单;校验:python3 scripts/fwd_verify_freeze.py",
        frozen_at=dict(local=local, utc=utc, source="date 命令"),
        git=dict(head=head, branch=branch, porcelain_counts=counts, porcelain_lines=len(porcelain.splitlines()),
                 porcelain_sha256=hashlib.sha256(porcelain.encode("utf-8")).hexdigest(),
                 porcelain_src_scripts_tests=focus),
        environment=_versions(),
        n_files=dict(core=sum(f["category"] == "core" for f in files), aux=sum(f["category"] == "aux" for f in files)),
        files=files,
        files_digest=digest,
        provenance=prov,
        prereg_params=PREREG_PARAMS,
        history=hist,
    )
    OUT.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_SHA.write_text("".join(f"{f['sha256']}  {f['path']}\n" for f in sorted(files, key=lambda x: x["path"])),
                       encoding="utf-8")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(local=local, utc=utc, files_digest=digest, n_files=len(files), git_head=head,
                                 manifest_sha256=sha256_file(OUT), reason=hist[-1]["reason"]),
                            ensure_ascii=False) + "\n")
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reason")
    a = ap.parse_args(argv)
    man = build(a.force, a.reason)
    print(f"冻结时间 {man['frozen_at']['local']}  git HEAD {man['git']['head'][:12]}  "
          f"文件 core {man['n_files']['core']} / aux {man['n_files']['aux']}")
    print(f"files_digest {man['files_digest']}")
    print(f"清单 sha256 {sha256_file(OUT)}")
    print(f"-> {OUT.relative_to(ROOT)}、{OUT_SHA.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
