---
pipeline_tag: tabular-classification
library_name: tabpfn
license: other
license_name: priorlabs-1-1
license_link: https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt
---

# TabPFN v2: A Tabular Foundation Model

TabPFN is a transformer-based foundation model for tabular data that leverages prior-data based learning to achieve strong performance on small tabular datasets without requiring task-specific training.

## Installation
```bash
pip install tabpfn
```

## Model Details
- **Developed by:** Prior Labs
- **Model type:** Transformer-based foundation model for tabular data
- **License:** [Prior Labs License (Apache 2.0 with additional attribution requirement)](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt)
- **Paper:** Published in Nature (January 2025)
- **Repository:** [GitHub - priorlabs/tabpfn](https://github.com/priorlabs/tabpfn)

### 📚 Citation

```bibtex
@article{hollmann2025tabpfn,
 title={Accurate predictions on small data with a tabular foundation model},
 author={Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and
         Krishnakumar, Arjun and K{\"o}rfer, Max and Hoo, Shi Bin and
         Schirrmeister, Robin Tibor and Hutter, Frank},
 journal={Nature},
 year={2025},
 month={01},
 day={09},
 doi={10.1038/s41586-024-08328-6},
 publisher={Springer Nature},
 url={https://www.nature.com/articles/s41586-024-08328-6},
}
```


## Quick Start

📚 For detailed usage examples and best practices, check out:
- [Interactive Colab Tutorial](https://tinyurl.com/tabpfn-colab-local)

## Technical Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.1
- scikit-learn ≥ 1.0
- Hardware: 16GB+ RAM, CPU (GPU optional)

## Checkpoints in this repository

This repository hosts the production TabPFN-v2 base checkpoints. Files matching the pattern `tabpfn-v2-classifier-finetuned-*.ckpt` are **content-identical aliases of the corresponding base checkpoints** (e.g. `tabpfn-v2-classifier-finetuned-gn2p4bpt-xp6f0iqb.ckpt` is identical to `tabpfn-v2-classifier-gn2p4bpt.ckpt`; `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` is identical to `tabpfn-v2-classifier-v2_default.ckpt`). The `finetuned-*` filenames are retained as aliases so that pinned filenames in user code continue to resolve.

## Resources
- **Documentation:** https://priorlabs.ai/docs
- **Source:** https://github.com/priorlabs/tabpfn
- **Paper:** https://doi.org/10.1038/s41586-024-08328-6

### Team
- Noah Hollmann
- Samuel Müller
- Lennart Purucker
- Arjun Krishnakumar
- Max Körfer
- Shi Bin Hoo
- Robin Tibor Schirrmeister
- Frank Hutter
- Eddie Bergman