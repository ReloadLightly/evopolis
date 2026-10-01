"""Public preregistration must contain the artifacts it promises to freeze."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from evopolis.institutional_data import (
    FROZEN_REQUIRED_FILES, _committed_sha256, _frozen_public_files, load_exp2,
)


class InstitutionalPublicFreezeTests(unittest.TestCase):
    def test_committed_blob_hash_reads_bounded_chunks_and_rejects_missing_blob(self):
        body = b'x' * ((2 << 20) + 17)
        reads = []

        class TrackingStream(io.BytesIO):
            def read(self, count=-1):
                reads.append(count)
                return super().read(count)

        process = Mock(stdout=TrackingStream(body))
        process.wait.return_value = 0
        process.poll.return_value = 0
        with patch('evopolis.institutional_data.subprocess.Popen', return_value=process):
            actual = _committed_sha256(Path('/tmp'), 'a'*40, 'forecast.json')
        self.assertEqual(actual, hashlib.sha256(body).hexdigest())
        self.assertTrue(reads and all(0 < size <= 1 << 20 for size in reads))

        def missing(*args, **kwargs):
            kwargs['stderr'].write(b'fatal: path not present in commit')
            failed = Mock(stdout=io.BytesIO())
            failed.wait.return_value = failed.poll.return_value = 128
            return failed

        with patch('evopolis.institutional_data.subprocess.Popen', side_effect=missing):
            with self.assertRaisesRegex(RuntimeError, 'commit lacks readable artifact forecast.json'):
                _committed_sha256(Path('/tmp'), 'a'*40, 'forecast.json')

    def test_local_valid_forecast_cannot_open_transfer_if_public_blob_is_wrong(self):
        for failure in ('missing', 'old'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                folder = root/'results/task06'
                folder.mkdir(parents=True)
                code = root/'analysis.py'
                code.write_bytes(b'# selected analysis\n')
                digest = hashlib.sha256(code.read_bytes()).hexdigest()
                forecast = 'results/task06/forecast.json'
                manifest = {'analysis_code_sha256': {'analysis.py': digest},
                            'forecast_summary_sha256': {forecast: digest},
                            'required_evidence_sha256': {'results/task06/verification.json': digest},
                            'checkpoints': {'family_17': {'path': 'weights/best.pt', 'sha256': digest}},
                            'seed_table': [{'trajectory_file': 'results/task06/forecast.npz', 'trajectory_sha256': digest}],
                            **{field: digest for field in FROZEN_REQUIRED_FILES.values()}}
                body = (json.dumps(manifest)+'\n').encode()
                (folder/'manifest.json').write_bytes(body)
                receipt = {'manifest_sha256': hashlib.sha256(body).hexdigest(),
                           'analysis_code_sha256': manifest['analysis_code_sha256'],
                           'commit': 'a'*40, 'remote': 'origin', 'ref': 'refs/heads/main'}
                (folder/'freeze_pushed.json').write_text(json.dumps(receipt))

                def git(args, **kwargs):
                    if args[1:3] == ['remote', 'get-url']:
                        return 'https://github.com/ReloadLightly/evopolis.git\n'
                    if args[1] == 'ls-remote':
                        return 'a'*40+'\trefs/heads/main\n'
                    if args[1] == 'show':
                        return body
                    raise AssertionError(args)

                expected = _frozen_public_files(manifest)

                def committed(root, commit, name):
                    if name == forecast:
                        if failure == 'missing':
                            raise RuntimeError('Experiment 2 is closed: frozen commit lacks readable artifact '+name)
                        return '0'*64
                    return expected[name]

                with patch('evopolis.institutional_data._verify_frozen_artifacts', return_value={'local': 'valid'}), \
                     patch('evopolis.institutional_data.subprocess.check_output', side_effect=git), \
                     patch('evopolis.institutional_data.subprocess.run', return_value=subprocess.CompletedProcess([], 0)), \
                     patch('evopolis.institutional_data._committed_sha256', side_effect=committed), \
                     patch('evopolis.institutional_data._parse_selected') as parser:
                    with self.assertRaisesRegex(RuntimeError, 'forecast.json'):
                        load_exp2(root/'never-read.csv', root=root)
                    parser.assert_not_called()

    def test_failed_human_parse_leaves_durable_first_opening_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = {'commit': 'a'*40, 'public_artifact_verification': {'all_committed_sha256_match': True}}
            with patch('evopolis.institutional_data.verify_freeze', return_value=receipt), \
                 patch('evopolis.institutional_data._source_check', return_value='b'*64), \
                 patch('evopolis.institutional_data._parse_selected', side_effect=ValueError('bad action')):
                with self.assertRaisesRegex(ValueError, 'bad action'):
                    load_exp2(root/'human.csv', root=root)
            marker = json.loads((root/'results/task06/experiment2_opening.json').read_text())
            self.assertEqual(marker['freeze_commit'], receipt['commit'])
            self.assertEqual(marker['source_sha256'], 'b'*64)
            self.assertEqual(marker['status'], 'parsing started; behavioral evidence may have been read')
            self.assertIn('first_access_utc', marker)


if __name__ == '__main__':
    unittest.main()
