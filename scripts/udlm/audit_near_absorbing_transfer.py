"""Fixed CPU-only extension of the previously pinned frozen-MDLM diagnostic."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.udlm import audit_mask_rich_transfer as original

MIXTURES = (0.9, 0.99, 0.999, 0.9999)
TIMES = (0.1, 0.5, 0.9)
SEED = 2400


def terminal_kl(probability, alpha=0.001):
    """Exact rank-one marginal KL to its positive prior at one clean token."""
    if not (0 < probability < 1 and 0 < alpha < 1):
        raise ValueError('This formula requires interior probabilities')
    retained = alpha + (1-alpha)*probability
    return retained*math.log(retained/probability) + (1-alpha)*(1-probability)*math.log1p(-alpha)


def main():
    previous = original.MIXTURES, original.TIMES, original.SEED
    buffer = io.StringIO()
    try:
        original.MIXTURES, original.TIMES, original.SEED = MIXTURES, TIMES, SEED
        with redirect_stdout(buffer):
            original.main()
    finally:
        original.MIXTURES, original.TIMES, original.SEED = previous
    result = json.loads(buffer.getvalue())
    panel = original.read_pinned(original.PANEL, original.PANEL_SHA)
    frequencies = original.read_pinned(original.FREQUENCY, original.FREQUENCY_SHA)
    expected = {'device': 'cpu', 'content_tokens': 814,
                'shape': [16, max(len(row['input_ids']) for row in panel['rows'][:16])],
                'checkpoint_sha256': original.CHECKPOINT_SHA,
                'panel_sha256': original.PANEL_SHA, 'frequency_sha256': original.FREQUENCY_SHA,
                'uniform_floor': 0.0002, 'noise_eps': 0.001,
                'inference_weights': {'source': 'MDLM EMA', 'ema_applied': True}}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError('Frozen diagnostic identity or shape differs from the declared design')
    slots = [(row['mask_mixture_weight'], row['time'], row['seed']) for row in result['results']]
    if slots != [(mixture, noise_time, SEED+index) for mixture in MIXTURES
                 for index, noise_time in enumerate(TIMES)]:
        raise ValueError('Frozen diagnostic condition/seed sequence differs from the declared design')
    counts = frequencies['counts_by_token_id']
    special = set(panel['tokenizer']['special_token_ids'])
    counts_tensor = original.torch.tensor(counts, dtype=original.torch.float64)
    base_tensor = (1-0.0002)*counts_tensor/counts_tensor.sum() + 0.0002/len(counts)
    base_tensor /= base_tensor.sum()
    base = base_tensor.tolist()
    # The pinned diagnostic checks that MASK is a special ID with zero count.
    # Every special has zero count here, hence the same smoothed base mass;
    # no unstored mask-token ID needs to be inferred from the panel metadata.
    assert special and all(counts[token] == 0 for token in special)
    base_mask_probability = base[min(special)]
    tokens = [token for row in panel['rows'][:16] for token in row['input_ids'] if token not in special]
    assert len(tokens) == result['content_tokens'] == 814
    for row in result['results']:
        mixture, noise_time = row['mask_mixture_weight'], row['time']
        row['one_token_tv_to_absorbing_marginal'] = 0.999*noise_time*(1-mixture)*(1-base_mask_probability)
    result['terminal_kl_over_observed_clean_content'] = [
        {'mask_mixture_weight': mixture, 'alpha_1': 0.001,
         'mean_nats': math.fsum(terminal_kl((1-mixture)*base[token]) for token in tokens)/len(tokens)}
        for mixture in MIXTURES
    ]
    result['study'] = 'near_absorbing_frozen_transfer_cpu'
    result['fixed_conditions'] = {'mixtures': MIXTURES, 'times': TIMES, 'base_seed': SEED,
                                  'forward_calls': len(MIXTURES)*len(TIMES), 'rows_per_forward': 16}
    for path in (Path(__file__), ROOT/'experiments/udlm/designs/near_absorbing_transfer_cpu.md'):
        result['source_hashes'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result['limitations'].append('Stronger MASK concentration changes corruption difficulty; no molecular benchmark inference follows.')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
