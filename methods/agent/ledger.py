#!/usr/bin/env python3
"""/lens 只追加账本（公开展示副本）。

  ledger.py append --file OUTPUT.json [--ledger-dir D]   # 追加一条，stdout 打印 entry_id
  ledger.py due [--date YYYY-MM-DD] [--ledger-dir D]     # 列出复查日期 ≤ date 的证伪条件（JSON）
  ledger.py show ENTRY_ID [--ledger-dir D]               # 打印该条记录（JSON）
  ledger.py verify [--ledger-dir D]                      # 重算整条哈希链

文件：<ledger-dir>/lens_ledger.jsonl，每行 {entry_id, appended_at, prev_hash, entry_hash, output}；
entry_hash = sha256(prev_hash + 规范化 output JSON)，首行 prev_hash = "GENESIS"；
entry_id = <lens>-<YYYYMMDDTHHMMSSZ>-<entry_hash 前 8 位>。
只以追加模式打开；追加前在独占锁下重算全链并校验链尾，链断即拒绝写入（退出码 3）。
中文渲染存到 <ledger-dir>/outputs/<entry_id>.md（OUTPUT.json 同目录有同名 .md 则复制，复制时把占位符
{{entry_id}} / {{ENTRY_ID}} 替换为真实 entry_id；没有占位符则在文末追加一行 entry_id；否则按 JSON 生成简版）。
jsonl 内的 output 原样保存，占位替换只影响 .md，哈希链不受影响。
校验：lens / question 缺失直接拒绝；其余契约字段缺失或枚举值偏离（sub_framework、gates.status、abstain.reason、
required_inputs.status、vetoes.triggered、asked_at 格式、未登记的顶层键、非弃权条目 falsifiers<2）只打警告。
--ledger-dir 默认取环境变量 LENS_LEDGER_DIR，否则为仓库内的 runtime/ledger；测试时务必设置其一。
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # 不在 skill 目录里留下 __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

LEDGER_NAME = 'lens_ledger.jsonl'
GENESIS = 'GENESIS'
LENSES = ('dalio', 'buffett', 'taleb')
ENTRY_ID_RE = re.compile(r'^(dalio|buffett|taleb)-(\d{8}T\d{6}Z)-([0-9a-f]{8})$')
RECOMMENDED_KEYS = ['schema_version', 'lens', 'card_version', 'question', 'asked_at', 'applicability', 'required_inputs',
                    'evidence_reading', 'gates', 'vetoes', 'verdict', 'confidence', 'abstain', 'falsifiers',
                    'conflicts_noted', 'quotes', 'snapshots', 'data_staleness_warning', 'disclaimer']
# CONTRACT §4 登记过的全部顶层键（含可选键）；其余顶层键只发警告，不拒绝
KNOWN_TOP_KEYS = set(RECOMMENDED_KEYS) | {'sub_framework', 'optional_inputs', 'routing', 'optional_inputs_missing'}
SUB_FRAMEWORKS = {'SF1', 'SF2', 'SF3', 'SF4', 'SF5', 'SF6', 'SF7', 'none'}
GATE_STATUSES = {'pass', 'fail', 'insufficient'}
ABSTAIN_REASONS = {'out_of_scope', 'missing_required_input', 'insufficient_data', 'timing_question', None}
INPUT_STATUSES = {'ok', 'stale', 'missing', 'proxy', 'user_required', 'user_provided'}
ASKED_AT_RE = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')
PLACEHOLDERS = ('{{entry_id}}', '{{ENTRY_ID}}')


class LedgerError(Exception):
    pass


def canonical(output) -> str:
    try:
        return json.dumps(output, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    except ValueError as exc:
        raise LedgerError(f'output 含 NaN/Infinity，无法规范化：{exc}')


def entry_hash(prev_hash: str, output) -> str:
    return hashlib.sha256((prev_hash + canonical(output)).encode('utf-8')).hexdigest()


def ledger_path(ledger_dir) -> Path:
    return Path(ledger_dir) / LEDGER_NAME


def parse_lines(raw: bytes):
    """返回 [(line_no, entry_dict | None, error | None)]。"""
    out = []
    if not raw:
        return out
    text = raw.decode('utf-8')
    lines = text.split('\n')
    if lines and lines[-1] == '':
        lines = lines[:-1]
    for i, line in enumerate(lines, 1):
        if not line.strip():
            out.append((i, None, '空行'))
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            out.append((i, None, f'不是合法 JSON：{exc}'))
            continue
        if not isinstance(obj, dict):
            out.append((i, None, '不是 JSON 对象'))
            continue
        out.append((i, obj, None))
    return out


def verify_bytes(raw: bytes):
    """重算全链。返回 (ok, errors, entries, head_hash)。"""
    errors = []
    entries = []
    prev = GENESIS
    if raw and not raw.endswith(b'\n'):
        errors.append('文件末尾缺少换行（最后一行可能写入不完整）')
    seen_ids = set()
    for line_no, obj, err in parse_lines(raw):
        if err:
            errors.append(f'第 {line_no} 行：{err}')
            prev = None
            continue
        missing = [k for k in ('entry_id', 'appended_at', 'prev_hash', 'entry_hash', 'output') if k not in obj]
        if missing:
            errors.append(f'第 {line_no} 行：缺字段 {missing}')
            prev = obj.get('entry_hash')
            continue
        if prev is not None and obj['prev_hash'] != prev:
            errors.append(f'第 {line_no} 行：prev_hash 与上一行 entry_hash 不符（链断）')
        try:
            recomputed = entry_hash(obj['prev_hash'], obj['output'])
        except LedgerError as exc:
            errors.append(f'第 {line_no} 行：{exc}')
            recomputed = None
        if recomputed != obj['entry_hash']:
            errors.append(f'第 {line_no} 行（{obj["entry_id"]}）：entry_hash 与内容不符（内容被改动）')
        m = ENTRY_ID_RE.match(str(obj['entry_id']))
        if not m:
            errors.append(f'第 {line_no} 行：entry_id 格式不符 {obj["entry_id"]}')
        else:
            if m.group(3) != str(obj['entry_hash'])[:8]:
                errors.append(f'第 {line_no} 行：entry_id 后缀与 entry_hash 前 8 位不符')
            if isinstance(obj['output'], dict) and m.group(1) != obj['output'].get('lens'):
                errors.append(f'第 {line_no} 行：entry_id 的 lens 与 output.lens 不符')
        if obj['entry_id'] in seen_ids:
            errors.append(f'第 {line_no} 行：entry_id 重复 {obj["entry_id"]}')
        seen_ids.add(obj['entry_id'])
        entries.append(obj)
        prev = obj['entry_hash']
    head = entries[-1]['entry_hash'] if entries else GENESIS
    return not errors, errors, entries, head


def read_ledger(ledger_dir):
    path = ledger_path(ledger_dir)
    if not path.exists():
        return b''
    with open(path, 'rb') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            return handle.read()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_output(output):
    if not isinstance(output, dict):
        raise LedgerError('OUTPUT.json 顶层必须是 JSON 对象')
    lens = output.get('lens')
    if lens not in LENSES:
        raise LedgerError(f'output.lens 必须是 {"/".join(LENSES)} 之一，收到 {lens!r}')
    if not str(output.get('question', '')).strip():
        raise LedgerError('output.question 不能为空')
    warnings = [f'缺少契约字段 {k}' for k in RECOMMENDED_KEYS if k not in output]
    extra = sorted(k for k in output if k not in KNOWN_TOP_KEYS)
    if extra:
        warnings.append(f'顶层含 CONTRACT §4 未登记的键 {extra}（不拒绝，但下游解析可能忽略）')
    # sub_framework：仅 dalio；枚举 SF1–SF7 / none
    sf = output.get('sub_framework')
    if lens == 'dalio':
        if not sf:
            warnings.append('dalio 输出缺 sub_framework')
        elif sf not in SUB_FRAMEWORKS:
            warnings.append(f'sub_framework 不在 SF1–SF7/none 之内：{sf!r}')
    elif sf not in (None, '', 'none'):
        warnings.append(f'非 dalio 输出不应填 sub_framework：{sf!r}')
    # asked_at：UTC ISO8601（Z 结尾）且不晚于当前时间
    asked = output.get('asked_at')
    if asked is not None:
        text = str(asked)
        if not ASKED_AT_RE.match(text):
            warnings.append(f'asked_at 不是 UTC ISO8601（YYYY-MM-DDTHH:MM:SSZ）：{text!r}')
        else:
            try:
                when = dt.datetime.strptime(text, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc)
                if when > C.utc_now() + dt.timedelta(minutes=5):
                    warnings.append(f'asked_at 晚于当前时间：{text}')
            except ValueError:
                warnings.append(f'asked_at 不是合法日期时间：{text!r}')
    # abstain
    ab = output.get('abstain')
    abstaining = False
    if isinstance(ab, dict):
        abstaining = bool(ab.get('is'))
        if ab.get('reason') not in ABSTAIN_REASONS:
            warnings.append(f'abstain.reason 不在枚举内（out_of_scope/missing_required_input/insufficient_data/timing_question/null）：{ab.get("reason")!r}')
        if abstaining and ab.get('reason') is None:
            warnings.append('abstain.is=true 但 reason 为空')
        if abstaining and not str(output.get('verdict', '')).startswith('弃权'):
            warnings.append('abstain.is=true 时 verdict 应以 "弃权：<abstain.reason>" 开头')
    elif ab is not None:
        warnings.append('abstain 不是对象')
    # confidence.value：0–1 的数；弃权时 null
    conf = output.get('confidence')
    if isinstance(conf, dict):
        cv = conf.get('value')
        if cv is not None and not (isinstance(cv, (int, float)) and not isinstance(cv, bool) and 0 <= cv <= 1):
            warnings.append(f'confidence.value 应为 0–1 的数或 null：{cv!r}')
        if abstaining and cv is not None:
            warnings.append('弃权条目 confidence.value 应为 null')
    # required_inputs
    for i, r in enumerate(output.get('required_inputs') or []):
        if not isinstance(r, dict):
            warnings.append(f'required_inputs[{i}] 不是对象')
            continue
        if r.get('status') not in INPUT_STATUSES:
            warnings.append(f'required_inputs[{i}].status 不在枚举内：{r.get("status")!r}')
        a = r.get('as_of')
        if a is not None:
            try:
                dt.date.fromisoformat(str(a))
            except ValueError:
                warnings.append(f'required_inputs[{i}].as_of 应为 YYYY-MM-DD 或 null，不写说明文字：{a!r}')
    # gates
    for i, g in enumerate(output.get('gates') or []):
        if not isinstance(g, dict):
            warnings.append(f'gates[{i}] 不是对象')
            continue
        if g.get('status') not in GATE_STATUSES:
            warnings.append(f'gates[{i}].status 不在 pass/fail/insufficient 内（未判定的关不要写进 gates）：{g.get("status")!r}')
    # vetoes：triggered 三态 true/false/null
    for i, v in enumerate(output.get('vetoes') or []):
        if isinstance(v, dict) and v.get('triggered') not in (True, False, None):
            warnings.append(f'vetoes[{i}].triggered 应为 true/false/null（null=未核）：{v.get("triggered")!r}')
    # falsifiers
    fals = output.get('falsifiers') or []
    if not abstaining and len(fals) < 2:
        warnings.append(f'非弃权条目 falsifiers 应至少 2 条，当前 {len(fals)} 条')
    for i, f in enumerate(fals):
        if not isinstance(f, dict):
            warnings.append(f'falsifiers[{i}] 不是对象')
            continue
        try:
            dt.date.fromisoformat(str(f.get('review_date', '')))
        except ValueError:
            warnings.append(f'falsifiers[{i}].review_date 不是 YYYY-MM-DD：{f.get("review_date")!r}')
        if abstaining and not str(f.get('condition', '')).startswith('复查触发'):
            warnings.append(f'弃权条目的 falsifiers[{i}].condition 应以 "复查触发：" 开头')
    canonical(output)  # NaN 检查
    return warnings


def render_markdown(entry_id, output) -> str:
    """没有同名 .md 时的简版中文渲染。"""
    def g(key, default='—'):
        v = output.get(key)
        return default if v in (None, '', [], {}) else v

    lines = [f'# /lens {output.get("lens")} · {entry_id}', '',
             f'**问题**：{g("question")}', '', f'**结论**：{g("verdict")}', '']
    app = output.get('applicability') or {}
    lines += [f'**适用性**：{app.get("status", "—")}；{app.get("reason", "")}' + (f'；转给 {app["redirect"]}' if app.get('redirect') else ''), '']
    if output.get('sub_framework'):
        lines += [f'**子框架**：{output["sub_framework"]}', '']
    ab = output.get('abstain') or {}
    if ab.get('is'):
        lines += [f'**弃权**：{ab.get("reason")}（范围 {ab.get("scope")}）', '']
    lines += ['## 必需输入', '', '| 名称 | 状态 | 值 | 日期 | 来源 |', '|---|---|---|---|---|']
    for r in output.get('required_inputs') or []:
        lines.append(f'| {r.get("name")} | {r.get("status")} | {r.get("value")} | {r.get("as_of")} | {r.get("source")} |')
    lines += ['', '## 关卡', '', '| 关卡 | 名称 | 结果 | 发现 |', '|---|---|---|---|']
    for r in output.get('gates') or []:
        lines.append(f'| {r.get("id")} | {r.get("name")} | {r.get("status")} | {r.get("finding")} |')
    lines += ['', '## 否决条件', '']
    for r in output.get('vetoes') or []:
        t = r.get('triggered')
        state = '触发' if t is True else ('未触发' if t is False else '未核')
        lines.append(f'- {r.get("id")}：{state}；{r.get("evidence", "")}')
    lines += ['', '## 证伪条件', '', '| 条件 | 指标 | 阈值 | 复查日 |', '|---|---|---|---|']
    for r in output.get('falsifiers') or []:
        lines.append(f'| {r.get("condition")} | {r.get("indicator")} | {r.get("threshold")} | {r.get("review_date")} |')
    conf = output.get('confidence') or {}
    lines += ['', f'**置信度**：{conf.get("value", "—")}（最不确定：{conf.get("least_certain_gate", "—")}）{conf.get("note", "")}', '',
              f'**数据陈旧提示**：{g("data_staleness_warning")}', '', f'entry_id：`{entry_id}`  ', f'{g("disclaimer", "第三人称框架推演，非本人观点；不构成投资建议")}', '']
    return '\n'.join(lines)


def cmd_append(args):
    src = Path(args.file)
    try:
        output = json.loads(src.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerError(f'读取 {src} 失败：{exc}')
    for w in validate_output(output):
        print(f'[ledger] 警告：{w}', file=sys.stderr)
    folder = Path(args.ledger_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = ledger_path(folder)
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)  # 追加模式：写入只会落在文件末尾
    with os.fdopen(fd, 'a+b') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            raw = handle.read()
            ok, errors, entries, head = verify_bytes(raw)
            if not ok:
                raise LedgerError('账本校验失败，拒绝追加：' + '；'.join(errors[:5]))
            body = canonical(output)
            if any(canonical(e['output']) == body for e in entries):
                print('[ledger] 警告：账本里已有完全相同的 output，本次仍按新条目登记', file=sys.stderr)
            now = C.utc_now()
            digest = entry_hash(head, output)
            entry_id = f'{output["lens"]}-{C.stamp_utc(now)}-{digest[:8]}'
            entry = {'entry_id': entry_id, 'appended_at': C.iso_utc(now), 'prev_hash': head,
                     'entry_hash': digest, 'output': output}
            line = json.dumps(entry, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'
            handle.write(line.encode('utf-8'))
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    outputs = folder / 'outputs'
    outputs.mkdir(exist_ok=True)
    md_src = src.with_suffix('.md')
    md_dst = outputs / f'{entry_id}.md'
    if md_src.exists():
        text = md_src.read_text(encoding='utf-8')
        if any(p in text for p in PLACEHOLDERS):
            for p in PLACEHOLDERS:
                text = text.replace(p, entry_id)
            how = f'复制自 {md_src}（已替换 entry_id 占位符）'
        else:
            text = text.rstrip('\n') + f'\n\nentry_id：`{entry_id}`  \n'
            how = f'复制自 {md_src}（无占位符，已在文末追加 entry_id）'
        content = text.encode('utf-8')
    else:
        content = render_markdown(entry_id, output).encode('utf-8')
        how = '按 JSON 自动生成'
    with open(md_dst, 'xb') as handle:
        handle.write(content)
    print(f'[ledger] 已追加到 {path}；渲染：{md_dst}（{how}）', file=sys.stderr)
    print(entry_id)
    return 0


def cmd_verify(args):
    path = ledger_path(args.ledger_dir)
    raw = read_ledger(args.ledger_dir)
    ok, errors, entries, head = verify_bytes(raw)
    md_missing = [e['entry_id'] for e in entries
                  if not (Path(args.ledger_dir) / 'outputs' / f'{e["entry_id"]}.md').exists()]
    result = {'status': 'ok' if ok else 'broken', 'ledger': str(path), 'exists': path.exists(),
              'entries': len(entries), 'head_hash': head, 'errors': errors}
    if md_missing:
        result['warnings'] = [f'缺少中文渲染 outputs/{x}.md' for x in md_missing]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def cmd_due(args):
    when = dt.date.fromisoformat(args.date) if args.date else C.today()
    raw = read_ledger(args.ledger_dir)
    ok, errors, entries, _ = verify_bytes(raw)
    items, invalid = [], []
    for e in entries:
        out = e.get('output') or {}
        for i, f in enumerate(out.get('falsifiers') or []):
            if not isinstance(f, dict):
                continue
            try:
                rd = dt.date.fromisoformat(str(f.get('review_date')))
            except ValueError:
                invalid.append({'entry_id': e['entry_id'], 'index': i, 'review_date': f.get('review_date')})
                continue
            if rd <= when:
                items.append({'entry_id': e['entry_id'], 'lens': out.get('lens'), 'question': out.get('question'),
                              'asked_at': out.get('asked_at'), 'falsifier_index': i, 'condition': f.get('condition'),
                              'indicator': f.get('indicator'), 'threshold': f.get('threshold'),
                              'review_date': rd.isoformat(), 'days_overdue': (when - rd).days})
    items.sort(key=lambda x: (x['review_date'], x['entry_id']))
    result = {'date': when.isoformat(), 'chain_ok': ok, 'count': len(items), 'items': items}
    if invalid:
        result['invalid_review_dates'] = invalid
    if not ok:
        result['chain_errors'] = errors
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_show(args):
    raw = read_ledger(args.ledger_dir)
    ok, errors, entries, _ = verify_bytes(raw)
    hits = [e for e in entries if e['entry_id'] == args.entry_id]
    if not hits:
        hits = [e for e in entries if e['entry_id'].startswith(args.entry_id)]
    if len(hits) != 1:
        print(json.dumps({'status': 'not_found' if not hits else 'ambiguous', 'entry_id': args.entry_id,
                          'matches': [e['entry_id'] for e in hits]}, ensure_ascii=False, indent=2))
        return 1
    entry = hits[0]
    md = Path(args.ledger_dir) / 'outputs' / f'{entry["entry_id"]}.md'
    print(json.dumps({'chain_ok': ok, 'markdown_path': str(md) if md.exists() else None, **entry},
                     ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='/lens 只追加账本：append / due / show / verify（公开展示副本）',
                                     epilog='环境变量 LENS_TODAY=YYYY-MM-DD 可覆盖 due 的默认日期。')
    sub = parser.add_subparsers(dest='cmd', required=True)

    def add_dir(p):
        p.add_argument('--ledger-dir', default=str(C.DEFAULT_LEDGER_DIR),
                       help=f'账本目录（默认取 LENS_LEDGER_DIR，否则 {C.OFFICIAL_LEDGER_DIR}；演示时建议指向临时目录）')

    p = sub.add_parser('append', help='追加一条透镜输出，打印 entry_id')
    p.add_argument('--file', required=True,
                   help='透镜输出 JSON 文件（结构化研究输出）；同目录同名 .md 会被复制为渲染，{{entry_id}} 占位符会被替换')
    add_dir(p)
    p = sub.add_parser('due', help='列出复查日期 ≤ date 的证伪条件')
    p.add_argument('--date', help='YYYY-MM-DD（默认今天）')
    add_dir(p)
    p = sub.add_parser('show', help='打印某条记录')
    p.add_argument('entry_id', help='entry_id（可用唯一前缀）')
    add_dir(p)
    p = sub.add_parser('verify', help='重算整条哈希链')
    add_dir(p)
    args = parser.parse_args(argv)
    try:
        return {'append': cmd_append, 'due': cmd_due, 'show': cmd_show, 'verify': cmd_verify}[args.cmd](args)
    except LedgerError as exc:
        print(json.dumps({'status': 'error', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 3


if __name__ == '__main__':
    sys.exit(main())
