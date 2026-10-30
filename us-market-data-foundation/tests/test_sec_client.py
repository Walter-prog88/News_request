"""Offline SEC transport/provenance/history tests; no production lake writes."""
import hashlib
from copy import deepcopy
import json
import unittest
from unittest.mock import MagicMock, patch

import requests

from usmd import sec_client as sec


def response(payload=None, status=200, body=None, headers=None):
    result = MagicMock()
    result.status_code = status
    result.headers = headers or {}
    result.__enter__.return_value = result
    raw = json.dumps(payload).encode() if body is None else body
    result.iter_content.return_value = [raw]
    if status >= 400:
        result.raise_for_status.side_effect = requests.HTTPError(response=result)
    return result


def primary(history=None, recent=None):
    return {'cik': '320193', 'name': 'Apple Inc.', 'filings': {
        'recent': recent or {'accessionNumber': ['new'], 'acceptanceDateTime': ['2026-07-01T16:00:00Z'],
                             'filingDate': ['2026-07-01'], 'form': ['10-Q']},
        'files': history or []}}


def page(name='CIK0000320193-submissions-001.json', filing_to='2020-01-01'):
    return {'name': name, 'filingFrom': '2010-01-01', 'filingTo': filing_to}


class SecClientTests(unittest.TestCase):
    def setUp(self):
        self.session_patch = patch.object(sec.requests, 'Session')
        self.session_type = self.session_patch.start()
        self.session = self.session_type.return_value
        self.sleep_patch = patch.object(sec.time, 'sleep')
        self.sleep = self.sleep_patch.start()
        sec._last_request_started = None
        self.client = sec.SecClient(user_agent='OfflineTests/1.0')

    def tearDown(self):
        self.session_patch.stop()
        self.sleep_patch.stop()
        sec._last_request_started = None

    def test_source_hash_and_bytes_are_not_reserialized(self):
        raw = b'{ "cik": 320193, "facts": {} }\n'
        self.session.get.return_value = response(body=raw, headers={'ETag': 'version1'})
        data, meta = self.client.get_companyfacts('CIK320193')
        self.assertEqual(data['cik'], 320193)
        self.assertEqual(meta['raw_bytes'], raw)
        self.assertEqual(meta['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(meta['byte_count'], len(raw))
        self.assertEqual(meta['etag'], 'version1')
        self.assertTrue(meta['retrieved_at'].endswith('+00:00'))
        self.session.get.assert_called_once_with(
            'https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json',
            timeout=(10, 45.0), stream=True, allow_redirects=False)

    def test_403_never_retries_or_attempts_mapping_fallback(self):
        self.session.get.return_value = response(status=403)
        with self.assertRaises(sec.SecRefusalError):
            self.client.get_tickers()
        self.assertEqual(self.session.get.call_count, 1)

    def test_retry_429_then_503_preserves_bounded_attempts(self):
        self.session.get.side_effect = [response(status=429, headers={'Retry-After': '5'}),
                                       response(status=503), response(payload={'ok': True})]
        data, _ = self.client.fetch_json(sec.TICKERS_URL)
        self.assertTrue(data['ok'])
        self.assertEqual(self.session.get.call_count, 3)
        self.assertIn(unittest.mock.call(5.0), self.sleep.call_args_list)
        self.assertIn(unittest.mock.call(2.0), self.sleep.call_args_list)

    def test_network_timeouts_stop_after_three_attempts(self):
        self.session.get.side_effect = requests.Timeout('offline')
        with self.assertRaises(requests.Timeout):
            self.client.fetch_json(sec.TICKERS_URL)
        self.assertEqual(self.session.get.call_count, 3)

    def test_404_is_not_retried(self):
        self.session.get.return_value = response(status=404)
        with self.assertRaises(requests.HTTPError):
            self.client.fetch_json(sec.TICKERS_URL)
        self.assertEqual(self.session.get.call_count, 1)

    def test_rate_limit_is_shared_across_instances(self):
        self.session.get.return_value = response(payload={})
        with patch.object(sec.time, 'monotonic', side_effect=[10.0, 10.0, 10.1, 10.5]):
            self.client.fetch_json(sec.TICKERS_URL)
            sec.SecClient().fetch_json(sec.TICKERS_URL)
        self.assertAlmostEqual(self.sleep.call_args.args[0], .4)

    def test_history_keeps_acceptance_times_columns_and_per_document_bytes(self):
        current = primary([page()])
        history = {'accessionNumber': ['new', 'old'],
                   'acceptanceDateTime': ['2026-07-01T15:59:00Z', '2017-01-25T16:21:00Z'],
                   'form': ['10-Q', '10-K'], 'primaryDocument': ['new.htm', 'old.htm']}
        self.session.get.side_effect = [response(current), response(history)]
        data, meta = self.client.get_submissions(320193, since='2016-01-01')
        recent = data['filings']['recent']
        self.assertEqual(recent['accessionNumber'], ['new', 'old'])
        self.assertEqual(recent['acceptanceDateTime'], ['2026-07-01T16:00:00Z', '2017-01-25T16:21:00Z'])
        self.assertEqual(recent['filingDate'], ['2026-07-01', None])
        self.assertEqual(recent['primaryDocument'], [None, 'old.htm'])
        self.assertEqual(meta['filing_count'], 2)
        self.assertEqual(meta['history_files_loaded'], 1)
        self.assertEqual(len(meta['documents']), 2)
        self.assertEqual(json.loads(meta['documents'][0]['raw_bytes']), current)
        self.assertEqual(json.loads(meta['documents'][1]['raw_bytes']), history)
        self.assertEqual(meta['sha256'], meta['documents'][0]['sha256'])

    def test_history_date_pruning_downloads_only_intersecting_pages(self):
        self.session.get.return_value = response(primary([page(filing_to='2015-12-31')]))
        _, meta = self.client.get_submissions('320193', since='2016-01-01')
        self.assertEqual(self.session.get.call_count, 1)
        self.assertEqual(meta['history_files_advertised'], 1)
        self.assertEqual(meta['history_files_loaded'], 0)

    def test_history_can_be_disabled(self):
        self.session.get.return_value = response(primary([page()]))
        _, meta = self.client.get_submissions(320193, include_history=False)
        self.assertEqual(self.session.get.call_count, 1)
        self.assertFalse(meta['history_included'])

    def test_failed_history_page_does_not_claim_complete_merge(self):
        self.session.get.side_effect = [response(primary([page()])), response(status=404)]
        with self.assertRaises(requests.HTTPError):
            self.client.get_submissions(320193)

    def test_unsafe_history_filename_is_rejected_before_request(self):
        self.session.get.return_value = response(primary([page(name='../bad.json')]))
        with self.assertRaisesRegex(ValueError, 'filename'):
            self.client.get_submissions(320193)
        self.assertEqual(self.session.get.call_count, 1)

    def test_malformed_column_lengths_fail_instead_of_truncating(self):
        self.session.get.return_value = response(primary(recent={'accessionNumber': ['a', 'b'], 'form': ['10-K']}))
        with self.assertRaisesRegex(ValueError, 'inconsistent lengths'):
            self.client.get_submissions(320193)

    def test_company_mismatch_is_rejected(self):
        self.session.get.return_value = response({'cik': 123, 'facts': {}})
        with self.assertRaisesRegex(ValueError, 'entity'):
            self.client.get_companyfacts(320193)

    def test_cik_and_url_validation_precede_network(self):
        for cik in (True, 0, '0', -1, 1.5, '12345678901', '320193/../../evil'):
            with self.subTest(cik=cik), self.assertRaises(ValueError):
                self.client.get_companyfacts(cik)
        for url in ('http://data.sec.gov/a', 'https://bad.example/a', 'https://user@data.sec.gov/a'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.client.fetch_json(url)
        self.session.get.assert_not_called()

    def test_unexpected_redirect_is_not_followed(self):
        self.session.get.return_value = response(status=302)
        with self.assertRaisesRegex(ValueError, 'redirect'):
            self.client.fetch_json(sec.TICKERS_URL)
        self.assertEqual(self.session.get.call_count, 1)

    def test_response_size_cap_applies_while_streaming(self):
        self.session.get.return_value = response(body=b'{"ok":true}')
        with patch.object(sec, 'MAX_RESPONSE_BYTES', 5), self.assertRaisesRegex(ValueError, 'byte limit'):
            self.client.fetch_json(sec.TICKERS_URL)

    def test_reconstruct_submissions_matches_fetch_merge_without_mutating_inputs(self):
        main = primary([page()])
        history = {'accessionNumber': ['new', 'old'], 'form': ['10-Q/A', '10-K'],
                   'acceptanceDateTime': ['conflicting', '2017-01-01T17:00:00Z'],
                   'primaryDocument': ['wrong.htm', 'historical.htm']}
        originals = deepcopy([main, history])
        reconstructed = sec.merge_submissions_documents([main, history])
        self.assertEqual([main, history], originals)
        self.session.get.side_effect = [response(main), response(history)]
        fetched, _ = self.client.get_submissions(320193)
        self.assertEqual(reconstructed, fetched)
        self.assertEqual(reconstructed['filings']['recent']['form'], ['10-Q', '10-K'])
        self.assertEqual(reconstructed['filings']['recent']['primaryDocument'], [None, 'historical.htm'])
        reconstructed['filings']['files'][0]['name'] = 'mutated.json'
        self.assertEqual([main, history], originals)

    def test_reconstruct_submissions_rejects_inconsistent_history(self):
        with self.assertRaisesRegex(ValueError, 'inconsistent lengths'):
            sec.merge_submissions_documents([primary(), {'accessionNumber': ['old'], 'form': []}])
        with self.assertRaises(ValueError):
            sec.merge_submissions_documents([])


if __name__ == '__main__':
    unittest.main()
