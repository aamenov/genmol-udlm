# V15 CPU export attempt 2: architecture type mismatch

Attempt 2 used the reviewed unused-configuration-resolution repair, exporter
`7b57a86b335d993e1a9500642f04fe999246be97`, wrapper
`e0b581e08c72b5c3b19f9f4834516dd196b45023`, and panel SHA-256
`2989c27799d3eda25999d78178b64fb3b7c9498814cfaaaef049bdeab6204c07`.
It stopped at the first `r099` export at 14:54:28.807919 UTC on 2026-09-07.
The remaining three entries were unlaunched. The first child ran 42.451108
seconds, including imports; the exporter recorded 29.672322 seconds and the
wrapper 46.584576 seconds. The owned process group exited without signals.

Source and target models were constructed on CPU, but exact BERT configuration
validation rejected the target before checkpoint serialization. The fixed export
JSON had `layer_norm_eps` as the string `"1e-12"`; the source checkpoint has the
numeric value `1e-12`. A separate CPU metadata-only checkpoint read and
`BertConfig.from_dict` comparison established this as the sole configuration
difference. The diagnostic constructed no models and performed no forward pass.
Neither operation performed GPU work, optimizer updates, molecular generation,
or property scoring. No derived checkpoint was produced.

The failed terminal SHA-256 is
`70c93cd1637824a2a661b1a87f0a55891830f3311d48555547ed4fb9e551c6a7`;
the architecture diagnostic SHA-256 is
`66deac8e0217eb610f06671636e4867582015fa4b2801e8aa0336438394e4a80`.
All six run/request/log receipts, the wrapper log, and the diagnostic are preserved
byte-for-byte. The architecture guard remains unchanged. A new export attempt
must pin corrected numeric epsilon values in all four configs and use a fresh
namespace, retaining the intended source architecture, priors, interpretations,
initialization seed, and original 14-run prospective molecular panel.
