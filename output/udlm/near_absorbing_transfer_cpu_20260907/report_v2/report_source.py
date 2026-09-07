"""Render the saved CPU token diagnostic; performs no model/oracle evaluation."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(data):
    return (json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def read(path, expected):
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError(f'Input hash differs: {path}')
    return data, {'path': str(path.resolve()), 'sha256': sha(data), 'size_bytes': len(data)}


def build(data, terminal):
    if (data['study'] != 'near_absorbing_frozen_transfer_cpu' or data['device'] != 'cpu'
            or data['shape'] != [16,106] or data['content_tokens'] != 814
            or terminal['status'] != 'completed' or terminal['return_code'] != 0
            or terminal['source'] != data['source_revision']):
        raise ValueError('Unexpected diagnostic identity/completion')
    expected = [(w,t,2400+i) for w in (.9,.99,.999,.9999) for i,t in enumerate((.1,.5,.9))]
    if [(r['mask_mixture_weight'],r['time'],r['seed']) for r in data['results']] != expected:
        raise ValueError('Fixed twelve-condition sequence differs')
    rows=[]
    for result in data['results']:
        groups = result['groups']
        if groups['all_content']['tokens'] != 814 or sum(groups[k]['tokens'] for k in groups if k!='all_content') != 814:
            raise ValueError('Token group partition differs')
        for group, values in groups.items():
            n = values['tokens']
            if n and (not math.isclose(values['ce_sum']/n, values['ce_mean'], rel_tol=2e-7, abs_tol=1e-6)
                      or values['correct_top1']/n != values['top1_accuracy']):
                raise ValueError('Saved group arithmetic differs')
            if not n and any(values[k] is not None for k in ('ce_mean','top1_accuracy','mean_unreduced_ce_logit_gradient_l2')):
                raise ValueError('Empty groups must retain null means')
            rows.append({'lambda':result['mask_mixture_weight'],'time':result['time'],
                         'seed':result['seed'],'group':group,**values,
                         'one_token_tv_to_absorbing':result['one_token_tv_to_absorbing_marginal']})
    csv_io=io.StringIO(newline='')
    writer=csv.DictWriter(csv_io,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    styles=getSampleStyleSheet()
    story=[]
    def para(text,style='BodyText'):
        story.append(Paragraph(text,styles[style]));story.append(Spacer(1,7))
    def table(cells,widths):
        cells=[[Paragraph(str(c),styles['BodyText']) for c in row] for row in cells]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6edf5')),
                               ('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.grey),
                               ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]))
        story.append(t);story.append(Spacer(1,9))
    para('Frozen MDLM: near-absorbing token diagnostic','Title')
    para('CPU diagnostic only. Zero training updates, generated molecules, property-oracle calls or GPUs. '
         'This is not a molecular quality result or evidence of GenMol superiority.')
    para('Same local MDLM50k EMA; 16 reused validation rows, 814 content tokens, padded shape [16,106]. '
         'All four MASK mixture weights and three noise times were declared before execution. '
         'Corruption seeds 2400/2401/2402 reset across priors at each time; twelve frozen forwards. '
         f"Four intra-op / one inter-op CPU threads. Producer runtime {data['runtime_seconds']:.3f}s; "
         f"complete child wall time {terminal['wall_seconds']:.3f}s.")
    cells=[['MASK weight','Time','All CE','Top-1 %','MASK n / CE','Changed other n / CE','Unchanged n / CE']]
    for r in data['results']:
        g=r['groups']
        def group(name):
            row=g[name]
            return str(row['tokens'])+' / '+('NA' if row['ce_mean'] is None else f"{row['ce_mean']:.3f}")
        cells.append([r['mask_mixture_weight'],r['time'],f"{g['all_content']['ce_mean']:.3f}",
                      f"{100*g['all_content']['top1_accuracy']:.2f}",group('currently_mask'),
                      group('changed_nonmask'),group('unchanged')])
    table(cells,[51,33,49,48,89,102,102])
    high_noise=[r for r in data['results'] if r['time']==0.9]
    first,last=high_noise[0]['groups'],high_noise[-1]['groups']
    para('CE is mean negative log probability of the original clean token (nats); top-1 is token accuracy. '
         f"At t=0.9, all-content CE changes from {first['all_content']['ce_mean']:.3f} at weight0.9 "
         f"to {last['all_content']['ce_mean']:.3f} at0.9999. Changed non-MASK observations change "
         f"from {first['changed_nonmask']['tokens']} to {last['changed_nonmask']['tokens']}. The corruption task changes, "
         'and empty/small groups cannot establish reliable conditional performance. '
         'All48 group records, CE sums and unreduced logit-gradient norms remain in groups.csv and the raw JSON.')
    story.append(PageBreak())
    para('Analytic comparison and interpretation','Heading2')
    para('Let pi_E be the smoothed empirical prior, m the MASK ID, lambda its mixture weight, '
         'and delta_j the point mass at clean token j. pi_lambda=lambda*delta_m+(1-lambda)*pi_E; '
         'q_t(.|j)=alpha_t*delta_j+(1-alpha_t)*pi_lambda, with alpha_t=1-0.999*t. '
         'The one-token total variation distance to the absorbing marginal is exactly '
         '(1-alpha_t)*(1-lambda)*(1-pi_E[m]), regardless of j. All tested priors have positive support.')
    table([['MASK weight','Mean terminal KL (nats)']]+[
        [r['mask_mixture_weight'],f"{r['mean_nats']:.9f}"] for r in data['terminal_kl_over_observed_clean_content']], [130,220])
    para('Terminal KL averages KL(q_1(.|j)||pi_lambda) over the814 observed clean tokens, with alpha_1=0.001. '
         'It increases as the prior approaches MASK and diverges at the absorbing limit for fixed positive alpha_1. '
         'Closer forward marginals alone do not establish equivalent learned reverse trajectories. '
         'The model was trained with MDLM masked-target loss; its unrestricted logits at unmasked observations '
         'need not be a calibrated clean posterior for the new categorical corruption. No CE-to-LOO conversion '
         'or reverse generation was performed here.')
    para('This extends an earlier diagnostic on the same reused rows. Common RNG streams are not maximal or '
         'nested coupling. One corruption draw per condition supplies no confidence interval. '
         'Mixed absorbing transitions are established in D3PM Appendix A.2.6 '
         '(https://arxiv.org/html/2107.03006). Any zero-update transfer or subsequent training needs a separate '
         'checkpoint identity and prospective molecular benchmark.')
    para('Provenance','Heading2')
    para('Producer source: '+data['source_revision']+'<br/>MDLM checkpoint SHA-256: '+data['checkpoint_sha256']+
         '<br/>Validation panel SHA-256: '+data['panel_sha256']+
         '<br/>Token frequency SHA-256: '+data['frequency_sha256'])
    pdf=io.BytesIO()
    SimpleDocTemplate(pdf,pagesize=A4,rightMargin=35,leftMargin=35,topMargin=30,bottomMargin=30,
                      title='Frozen MDLM near-absorbing CPU diagnostic',invariant=1).build(story)
    return {'groups.csv':csv_io.getvalue().encode(),'report.pdf':pdf.getvalue()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','terminal','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('input-sha256','terminal-sha256'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    raw, raw_claim=read(args.input,args.input_sha256)
    terminal_raw, terminal_claim=read(args.terminal,args.terminal_sha256)
    terminal=json.loads(terminal_raw)
    if terminal['result_sha256']!=sha(raw):raise ValueError('Terminal result identity differs')
    files=build(json.loads(raw),terminal)
    source=Path(__file__).read_bytes()
    files['report_source.py']=source
    inputs=[raw_claim,terminal_claim]
    for claim in inputs:read(Path(claim['path']),claim['sha256'])
    manifest={'schema_version':1,'study':'near_absorbing_frozen_transfer_cpu','inputs':inputs,
              'source_sha256':sha(source),'outputs':{name:{'sha256':sha(data),'size_bytes':len(data)} for name,data in files.items()},
              'metric_scope':'token diagnostic, distinct from molecular benchmarks; no new evaluations'}
    files['manifest.json']=encode(manifest)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,data in files.items():
        with (args.output/name).open('xb') as stream:stream.write(data)
    print(json.dumps({'output':str(args.output.resolve()),'pdf_sha256':manifest['outputs']['report.pdf']['sha256']}))


if __name__=='__main__':main()
