#!/usr/bin/env python3
"""Synthetic ledger checks; only temporary directories are written.
Run: python -m unittest discover -s agent -p 'test_ledger.py' -v
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / 'ledger.py'


def minimal_output(lens, question, review_dates, verdict='信息不足'):
    out = {
        'schema_version': '1', 'lens': lens, 'card_version': f'{lens}-v1-2026-10-03',
        'question': question, 'asked_at': '2026-10-03T12:00:00Z',
        'applicability': {'status': 'applicable', 'reason': '测试', 'redirect': ''},
        'required_inputs': [{'name': 'exit_line', 'status': 'user_provided', 'value': 40, 'as_of': '2026-10-03', 'source': 'user'}],
        'evidence_reading': '测试读数', 'gates': [{'id': 'G1', 'name': '测试关', 'status': 'insufficient', 'finding': '—', 'evidence': []}],
        'vetoes': [{'id': 'V1', 'triggered': False, 'evidence': '—'}], 'verdict': verdict,
        'confidence': {'value': 0.5, 'least_certain_gate': 'G1', 'note': '测试'},
        'abstain': {'is': False, 'scope': [], 'reason': None},
        'falsifiers': [{'condition': f'条件{i}', 'indicator': 'vix', 'threshold': '>30', 'review_date': d}
                       for i, d in enumerate(review_dates)],
        'conflicts_noted': [], 'quotes': [], 'snapshots': [], 'data_staleness_warning': '',
        'disclaimer': '第三人称框架推演，非本人观点；不构成投资建议',
    }
    if lens == 'dalio':
        out['sub_framework'] = 'SF1'
    return out


def abstain_output(lens, question):
    """CONTRACT §4 弃权条目填法。"""
    out = minimal_output(lens, question, [])
    out.update({
        'applicability': {'status': 'not_applicable', 'reason': '测试：时点问题', 'redirect': 'Taleb'},
        'required_inputs': [], 'gates': [], 'vetoes': [{'id': 'V3', 'triggered': True, 'evidence': '时点问题'}],
        'verdict': '弃权：timing_question',
        'confidence': {'value': None, 'least_certain_gate': None, 'note': '弃权，无结论'},
        'abstain': {'is': True, 'scope': ['整体'], 'reason': 'timing_question'},
        'falsifiers': [], 'snapshots': [], 'data_staleness_warning': '未取数',
    })
    if lens == 'dalio':
        out['sub_framework'] = 'none'
    return out


class LedgerTest(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / 'runtime' / 'tests'
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='lens_ledger_test_', dir=temporary_root)
        self.dir = Path(self.tmp.name) / 'ledger'
        self.work = Path(self.tmp.name) / 'work'
        self.work.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def run_ledger(self, *args, env=None):
        full_env = dict(os.environ, **(env or {}))
        return subprocess.run([sys.executable, str(LEDGER), *args, '--ledger-dir', str(self.dir)],
                              capture_output=True, text=True, env=full_env)

    def append(self, name, output, md=None, expect_warning=None, forbid_warning=None):
        path = self.work / f'{name}.json'
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
        if md is not None:
            path.with_suffix('.md').write_text(md, encoding='utf-8')
        res = self.run_ledger('append', '--file', str(path))
        self.assertEqual(res.returncode, 0, res.stderr)
        if expect_warning:
            self.assertIn(expect_warning, res.stderr)
        if forbid_warning:
            self.assertNotIn(forbid_warning, res.stderr)
        return res.stdout.strip()

    def lines(self):
        return (self.dir / 'lens_ledger.jsonl').read_text(encoding='utf-8').splitlines()

    def test_append_verify_tamper(self):
        ids = [
            self.append('a', minimal_output('taleb', '一次 40% 下跌会不会让我出局', ['2026-11-01', '2027-01-15'])),
            self.append('b', minimal_output('buffett', '虚构公司 EXAMPLE 生意质量', ['2026-12-31']), md='# 手写渲染\n内容\n'),
            self.append('c', minimal_output('dalio', '短债周期位置', ['2026-10-01'])),
        ]
        for i, lens in zip(ids, ('taleb', 'buffett', 'dalio')):
            self.assertRegex(i, rf'^{lens}-\d{{8}}T\d{{6}}Z-[0-9a-f]{{8}}$')
        lines = self.lines()
        self.assertEqual(len(lines), 3)
        entries = [json.loads(x) for x in lines]
        self.assertEqual(entries[0]['prev_hash'], 'GENESIS')
        for prev, cur in zip(entries, entries[1:]):
            self.assertEqual(cur['prev_hash'], prev['entry_hash'])
        # entry_hash = sha256(prev_hash + 规范化 output)
        e = entries[1]
        canon = json.dumps(e['output'], sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        self.assertEqual(e['entry_hash'], hashlib.sha256((e['prev_hash'] + canon).encode('utf-8')).hexdigest())
        self.assertTrue(e['entry_id'].endswith(e['entry_hash'][:8]))
        # 渲染：b 复制手写 .md（无占位符 → 文末追加 entry_id 行），a/c 自动生成
        md_b = (self.dir / 'outputs' / f'{ids[1]}.md').read_text(encoding='utf-8')
        self.assertTrue(md_b.startswith('# 手写渲染\n内容\n'))
        self.assertIn(f'entry_id：`{ids[1]}`', md_b)
        self.assertIn(ids[0], (self.dir / 'outputs' / f'{ids[0]}.md').read_text(encoding='utf-8'))

        res = self.run_ledger('verify')
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertEqual(json.loads(res.stdout)['entries'], 3)

        due = json.loads(self.run_ledger('due', '--date', '2026-12-31').stdout)
        self.assertEqual(due['count'], 3)  # 2026-10-01、2026-11-01、2026-12-31；2027-01-15 未到期
        self.assertEqual([x['review_date'] for x in due['items']], ['2026-10-01', '2026-11-01', '2026-12-31'])
        self.assertTrue(due['chain_ok'])
        due_env = json.loads(self.run_ledger('due', env={'LENS_TODAY': '2026-10-03'}).stdout)
        self.assertEqual(due_env['count'], 1)

        shown = self.run_ledger('show', ids[2])
        self.assertEqual(shown.returncode, 0)
        self.assertEqual(json.loads(shown.stdout)['output']['lens'], 'dalio')
        self.assertEqual(self.run_ledger('show', 'taleb-19990101T000000Z-deadbeef').returncode, 1)

        # 篡改第 2 行的 output（JSON 仍合法）
        ledger_file = self.dir / 'lens_ledger.jsonl'
        tampered = json.loads(lines[1])
        tampered['output']['verdict'] = '通过'
        lines[1] = json.dumps(tampered, ensure_ascii=False, separators=(',', ':'))
        ledger_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        res = self.run_ledger('verify')
        self.assertNotEqual(res.returncode, 0, '篡改后 verify 必须失败')
        report = json.loads(res.stdout)
        self.assertEqual(report['status'], 'broken')
        self.assertTrue(any('内容被改动' in x for x in report['errors']), report['errors'])

        # 链断后 append 必须拒绝，文件不变
        before = ledger_file.read_bytes()
        path = self.work / 'd.json'
        path.write_text(json.dumps(minimal_output('taleb', '再来一条', ['2027-01-01'])), encoding='utf-8')
        res = self.run_ledger('append', '--file', str(path))
        self.assertEqual(res.returncode, 3, res.stderr)
        self.assertEqual(ledger_file.read_bytes(), before)

    def test_deleted_line_breaks_chain(self):
        for i in range(3):
            self.append(f'x{i}', minimal_output('taleb', f'问题{i}', ['2027-01-01']))
        lines = self.lines()
        (self.dir / 'lens_ledger.jsonl').write_text(lines[0] + '\n' + lines[2] + '\n', encoding='utf-8')
        res = self.run_ledger('verify')
        self.assertNotEqual(res.returncode, 0)
        self.assertTrue(any('链断' in x for x in json.loads(res.stdout)['errors']))

    def test_rejects_invalid_lens(self):
        path = self.work / 'bad.json'
        bad = minimal_output('taleb', 'x', [])
        bad['lens'] = 'soros'
        path.write_text(json.dumps(bad), encoding='utf-8')
        res = self.run_ledger('append', '--file', str(path))
        self.assertEqual(res.returncode, 3)
        self.assertFalse((self.dir / 'lens_ledger.jsonl').exists())

    def test_empty_ledger_verifies(self):
        res = self.run_ledger('verify')
        self.assertEqual(res.returncode, 0)
        self.assertEqual(json.loads(res.stdout)['entries'], 0)

    def test_md_placeholder_and_abstain_rules(self):
        # 1) .md 里的 {{entry_id}} 占位符被替换；jsonl 内 output 不变
        out = minimal_output('taleb', '占位符测试', ['2027-01-01', '2027-02-01'])
        eid = self.append('p1', out, md='# 渲染\n\nentry_id：{{entry_id}}\n', forbid_warning='falsifiers 应至少 2 条')
        md = (self.dir / 'outputs' / f'{eid}.md').read_text(encoding='utf-8')
        self.assertIn(f'entry_id：{eid}', md)
        self.assertNotIn('{{entry_id}}', md)
        self.assertEqual(json.loads(self.lines()[0])['output'], out)
        # 2) 没有占位符 → 文末追加 entry_id 行
        eid2 = self.append('p2', minimal_output('buffett', '无占位符', ['2027-01-01', '2027-02-01']), md='# 渲染\n正文\n')
        md2 = (self.dir / 'outputs' / f'{eid2}.md').read_text(encoding='utf-8')
        self.assertTrue(md2.startswith('# 渲染\n正文\n'))
        self.assertIn(f'entry_id：`{eid2}`', md2)
        # 3) 弃权条目：falsifiers 为空不告警；非弃权条目 falsifiers<2 告警
        self.append('p3', abstain_output('dalio', '哪个月见顶'), forbid_warning='falsifiers 应至少 2 条')
        self.append('p4', minimal_output('taleb', '只有一条证伪', ['2027-01-01']), expect_warning='falsifiers 应至少 2 条')
        # 4) 枚举偏离只警告不拒绝
        bad = minimal_output('dalio', '枚举偏离', ['2027-01-01', '2027-02-01'])
        bad['sub_framework'] = '短期债务周期'
        bad['gates'][0]['status'] = 'not_run'
        bad['required_inputs'][0]['status'] = 'unknown'
        bad['asked_at'] = '2026-10-03 12:00'
        bad['extra_key'] = 1
        self.append('p5', bad, expect_warning='sub_framework 不在 SF1–SF7/none 之内')
        res = self.run_ledger('verify')
        self.assertEqual(res.returncode, 0, res.stdout)
        self.assertEqual(json.loads(res.stdout)['entries'], 5)
        # 5) 自动渲染：vetoes.triggered=null 显示"未核"
        tri = minimal_output('taleb', '三态否决', ['2027-01-01', '2027-02-01'])
        tri['vetoes'] = [{'id': 'V2', 'triggered': None, 'evidence': '缺 user:funding_terms'}]
        eid6 = self.append('p6', tri)
        self.assertIn('V2：未核', (self.dir / 'outputs' / f'{eid6}.md').read_text(encoding='utf-8'))

    def test_env_ledger_dir_default(self):
        """不传 --ledger-dir 时，LENS_LEDGER_DIR 决定默认目录（测试不得写正式账本）。"""
        env_dir = Path(self.tmp.name) / 'env_ledger'
        res = subprocess.run([sys.executable, str(LEDGER), 'verify'], capture_output=True, text=True,
                             env=dict(os.environ, LENS_LEDGER_DIR=str(env_dir)))
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(json.loads(res.stdout)['ledger'], str(env_dir / 'lens_ledger.jsonl'))


if __name__ == '__main__':
    unittest.main()
