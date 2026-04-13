# Data

## UNFAIR-ToS (from LexGLUE benchmark)
- **Source:** `coastalcph/lex_glue` on HuggingFace (config: `unfair_tos`)
- **Paper:** Chalkidis et al. (2022) — LexGLUE: A Benchmark Dataset for Legal Language Understanding (ACL 2022)
- **License:** CC BY-SA 4.0
- **Task:** Multi-label sentence classification — detect 8 types of potentially unfair clauses in Terms of Service documents
- **Size:** 9,414 sentences (5,532 train / 2,275 val / 1,607 test) from 50 ToS documents
- **Labels:** Limitation of liability, Unilateral termination, Unilateral change, Content removal, Contract by using, Choice of law, Jurisdiction, Arbitration

## Download
Data is loaded automatically via the `datasets` library:
```python
from datasets import load_dataset
ds = load_dataset('coastalcph/lex_glue', 'unfair_tos')
```
