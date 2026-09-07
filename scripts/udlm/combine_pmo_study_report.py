"""Preserve the 93-page V13 study while prepending a published V14 report bundle.

This checks saved-file identities and PDF preservation only. It never computes
molecular metrics, runs an oracle, reads a checkpoint, or changes report status.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter

BASE_PDF_SHA = '350d365017b5a87291d772389b8767d44094e40dd6cb1e99d30b37951b206fdf'
BASE_MANIFEST_SHA = 'db76e519891a1891b0587653d9f0c49602f814c739611c430ba800dc72108b20'
PANEL_SHA = '8f42ce172b392d27260317f3db7e07ce5c5cbe528eb448234388fcc34e0f4fda'
PMO_OUTPUTS = {'report.json', 'runs.csv', 'contrasts.csv', 'report.pdf', 'report_source.py'}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, expected=None):
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    actual = digest(payload)
    require(expected is None or actual == expected, f'Input SHA-256 differs: {path}')
    return {'path': str(path), 'sha256': actual, 'size_bytes': len(payload)}, payload


def pdf_signature(page):
    contents = page.get_contents()
    return (page.extract_text(), contents.get_data() if contents is not None else b'')


def combine_pages(front, previous):
    """Keep every page in order, including negative findings and failed studies."""
    readers = [PdfReader(io.BytesIO(data)) for data in (front, previous)]
    require(all(not reader.is_encrypted for reader in readers), 'Encrypted PDF unsupported')
    writer = PdfWriter()
    for reader in readers:
        for page in reader.pages:
            writer.add_page(page)
    writer.add_metadata({'/Title': 'GenMol / UDLM study through V14',
                         '/Subject': 'Fixed PMO pilot and all preserved prior study pages'})
    stream = io.BytesIO()
    writer.write(stream)
    payload = stream.getvalue()
    merged = PdfReader(io.BytesIO(payload))
    originals = [page for reader in readers for page in reader.pages]
    require(len(merged.pages) == len(originals), 'Combined page count differs')
    for original, copy in zip(originals, merged.pages):
        require(pdf_signature(original) == pdf_signature(copy), 'PDF page content changed')
    return payload, [len(reader.pages) for reader in readers]


def build(base_directory, pmo_directory, pmo_manifest_sha):
    require(isinstance(pmo_manifest_sha, str) and len(pmo_manifest_sha) == 64
            and all(character in '0123456789abcdef' for character in pmo_manifest_sha),
            'An explicit lowercase PMO manifest SHA-256 is required')
    base, pmo = Path(base_directory), Path(pmo_directory)
    inputs = []
    base_record, base_pdf = read(base / 'study_overview.pdf', BASE_PDF_SHA)
    inputs.append(base_record)
    base_manifest, _ = read(base / 'input_hash_manifest.json', BASE_MANIFEST_SHA)
    inputs.append(base_manifest)
    claim, manifest_bytes = read(pmo / 'manifest.json', pmo_manifest_sha)
    inputs.append(claim)
    manifest = json.loads(manifest_bytes)
    require(manifest.get('schema_version') == 1 and set(manifest['outputs']) == PMO_OUTPUTS,
            'Unexpected PMO publication schema/output set')
    saved = {}
    for name in sorted(PMO_OUTPUTS):
        record, payload = read(pmo / name, manifest['outputs'][name]['sha256'])
        require(record['size_bytes'] == manifest['outputs'][name]['size_bytes'],
                'PMO output size differs')
        inputs.append(record)
        saved[name] = payload
    require(manifest['source']['sha256'] == digest(saved['report_source.py']),
            'Archived PMO source differs from its manifest')
    report = json.loads(saved['report.json'])
    require(report.get('study') == 'V14' and report.get('task') == 'fexofenadine_mpo'
            and report.get('panel_sha256') == PANEL_SHA, 'Wrong fixed PMO report')
    require(report['status'] in {'verified_complete', 'pending_verification', 'incomplete_or_unrankable'},
            'Unknown report status')
    require(report['planned_runs'] == 6 and report['planned_online_calls'] == 12000,
            'Fixed PMO accounting differs')
    source, source_bytes = read(__file__)
    inputs.append(source)
    pdf, pages = combine_pages(saved['report.pdf'], base_pdf)
    require(pages[1] == 93 and 1 <= pages[0] <= 10, 'Unexpected original report page count')
    outputs = {'study_overview.pdf': pdf, 'combiner_source.py': source_bytes}
    result = {'schema_version': 1, 'inputs': inputs,
              'outputs': {name: {'sha256': digest(data), 'size_bytes': len(data)}
                          for name, data in outputs.items()},
              'packages': {'pypdf': importlib.metadata.version('pypdf')},
              'pmo_report_status': report['status'],
              'pages': {'pmo': pages[0], 'preserved_v13': pages[1], 'total': sum(pages)},
              'accounting': {'denovo_configurations': 34, 'denovo_runs': 68,
                             'denovo_accepted_requests': 5504, 'pmo_planned_runs': 6,
                             'pmo_planned_online_calls': 12000,
                             'pmo_completed_runs': report['completed_runs'],
                             'pmo_accepted_online_calls': report['accepted_online_calls'],
                             'pmo_verified_online_calls': report['verified_online_calls']},
              'preservation': 'Every source page retains identical extracted text and decoded content stream; source reports remain unchanged.',
              'audit_boundary': 'Validates externally pinned publication hashes and every PMO output. Earlier molecular audits are retained in the original reports; this combiner does not repeat them or promote a study.'}
    outputs['manifest.json'] = encode(result)
    # Recheck captured inputs after parsing and merging; publish only fixed bytes.
    for record in inputs:
        observed, _ = read(record['path'], record['sha256'])
        require(observed == record, 'Input changed during combination')
    return outputs, result


def publish(files, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Completion marker last: interrupted bundles have no manifest.json.
    for name in [*sorted(set(files) - {'manifest.json'}), 'manifest.json']:
        with (output / name).open('xb') as stream:
            stream.write(files[name])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-directory', type=Path, required=True)
    parser.add_argument('--pmo-directory', type=Path, required=True)
    parser.add_argument('--pmo-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    files, result = build(args.base_directory, args.pmo_directory, args.pmo_manifest_sha256)
    publish(files, args.output)
    print(json.dumps({'output': str(args.output.resolve()), 'pages': result['pages'],
                      'pmo_status': result['pmo_report_status'],
                      'pdf_sha256': result['outputs']['study_overview.pdf']['sha256']}))


if __name__ == '__main__':
    main()
