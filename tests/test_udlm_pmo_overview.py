import hashlib
import io
import json
from pathlib import Path

import pytest
from pypdf import PdfReader
from reportlab.pdfgen import canvas

from scripts.udlm import combine_pmo_study_report as combine

BASE = Path(__file__).resolve().parents[1] / 'output/udlm/study_overview_v13_20260907'


def synthetic_bundle(directory, status='pending_verification'):
    directory.mkdir()
    pdf = io.BytesIO()
    writer = canvas.Canvas(pdf, invariant=1)
    writer.drawString(35, 800, 'SYNTHETIC PMO: negative contrast -0.125; status=' + status)
    writer.showPage()
    writer.save()
    report = {'study': 'V14', 'task': 'fexofenadine_mpo', 'panel_sha256': combine.PANEL_SHA,
              'status': status, 'planned_runs': 6, 'planned_online_calls': 12000,
              'completed_runs': 6 if status != 'incomplete_or_unrankable' else 1,
              'accepted_online_calls': 12000 if status != 'incomplete_or_unrankable' else 2000,
              'verified_online_calls': 12000 if status == 'verified_complete' else 0}
    files = {'report.json': combine.encode(report), 'runs.csv': b'arm,value\nS,-0.125\n',
             'contrasts.csv': b'contrast,value\nS-minus-MDLM,-0.125\n',
             'report.pdf': pdf.getvalue(), 'report_source.py': b'# Synthetic report generator\n'}
    manifest = {'schema_version': 1, 'source': {'sha256': combine.digest(files['report_source.py'])},
                'outputs': {name: {'sha256': combine.digest(data), 'size_bytes': len(data)}
                            for name, data in files.items()}}
    for name, data in files.items():
        (directory / name).write_bytes(data)
    payload = combine.encode(manifest)
    (directory / 'manifest.json').write_bytes(payload)
    return combine.digest(payload)


@pytest.mark.parametrize('status', ['verified_complete', 'pending_verification', 'incomplete_or_unrankable'])
def test_preserves_original_93_pages_and_report_status(tmp_path, status):
    pmo = tmp_path / 'pmo'
    sha = synthetic_bundle(pmo, status)
    original = (BASE / 'study_overview.pdf').read_bytes()
    files, receipt = combine.build(BASE, pmo, sha)
    assert receipt['pages'] == {'pmo': 1, 'preserved_v13': 93, 'total': 94}
    assert receipt['pmo_report_status'] == status
    assert receipt['accounting']['pmo_verified_online_calls'] == (12000 if status == 'verified_complete' else 0)
    merged, old = PdfReader(io.BytesIO(files['study_overview.pdf'])), PdfReader(io.BytesIO(original))
    assert '-0.125' in merged.pages[0].extract_text()
    assert [combine.pdf_signature(page) for page in merged.pages[1:]] == [combine.pdf_signature(page) for page in old.pages]
    assert (BASE / 'study_overview.pdf').read_bytes() == original


def test_byte_determinism_and_exclusive_publication(tmp_path):
    pmo = tmp_path / 'pmo'
    sha = synthetic_bundle(pmo)
    first, _ = combine.build(BASE, pmo, sha)
    second, _ = combine.build(BASE, pmo, sha)
    assert first == second
    output = tmp_path / 'combined'
    combine.publish(first, output)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        combine.publish(second, output)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before == first


def test_corrupted_report_output_is_rejected(tmp_path):
    pmo = tmp_path / 'pmo'
    sha = synthetic_bundle(pmo)
    (pmo / 'contrasts.csv').write_bytes(b'altered negative result')
    with pytest.raises(ValueError, match='SHA-256 differs'):
        combine.build(BASE, pmo, sha)


@pytest.mark.parametrize('sha', [None, '', 'z' * 64, 'A' * 64])
def test_external_manifest_hash_required_before_reading(sha):
    with pytest.raises(ValueError, match='explicit lowercase'):
        combine.build('missing', 'missing', sha)


def test_wrong_manifest_hash_is_rejected(tmp_path):
    pmo = tmp_path / 'pmo'
    synthetic_bundle(pmo)
    with pytest.raises(ValueError, match='SHA-256 differs'):
        combine.build(BASE, pmo, hashlib.sha256(b'wrong manifest').hexdigest())
