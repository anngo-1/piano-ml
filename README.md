# pianogen

Train and sample a decoder-only Transformer that generates piano music as REMI tokens.

## Model Architecture

The model is a causal decoder-only Transformer over REMI event tokens. It predicts the next token, then decodes generated token streams back into MIDI.

Architecture details:

- 2048-token training/evaluation sequence length in the provided configs
- rotary position embeddings
- grouped-query self-attention
- PyTorch scaled dot-product attention
- RMSNorm
- SwiGLU feed-forward blocks
- tied token embedding / output projection
- KV-cached PyTorch autoregressive generation

Training/inference details:

- MAESTRO v3.0.0 MIDI data
- REMI tokenization
- Muon optimizer
- cosine learning-rate schedule
- optional full-sequence ONNX export path
- optional CPU int8 inference path in the app

## References

These are papers and projects that inspired this implementation. This repository is not a faithful reproduction of any one paper; it combines a decoder-only Transformer music model with REMI-style tokenization and modern Transformer components.

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) introduced the Transformer architecture.
- [Music Transformer](https://arxiv.org/abs/1809.04281) motivated Transformer-based symbolic music generation with long-range structure.
- [Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset](https://arxiv.org/abs/1810.12247) introduced the MAESTRO piano dataset used by this project.
- [Pop Music Transformer](https://arxiv.org/abs/2002.00212) introduced REMI-style beat-based event tokenization for expressive piano generation.
- [RoFormer](https://arxiv.org/abs/2104.09864) introduced rotary position embeddings.
- [GQA](https://aclanthology.org/2023.emnlp-main.298/) introduced grouped-query attention.
- [Root Mean Square Layer Normalization](https://papers.neurips.cc/paper/9403-root-mean-square-layer-normalization) introduced RMSNorm.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) introduced SwiGLU-style feed-forward variants.
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) introduced AdamW.
- [Muon](https://github.com/KellerJordan/Muon) is the optimizer implementation this project follows.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{vaswani2017attention,
  title = {Attention Is All You Need},
  author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {30},
  year = {2017},
  url = {https://arxiv.org/abs/1706.03762}
}

@inproceedings{huang2019music,
  title = {Music Transformer},
  author = {Huang, Cheng-Zhi Anna and Vaswani, Ashish and Uszkoreit, Jakob and Shazeer, Noam and Simon, Ian and Hawthorne, Curtis and Dai, Andrew M. and Hoffman, Matthew D. and Dinculescu, Monica and Eck, Douglas},
  booktitle = {International Conference on Learning Representations},
  year = {2019},
  url = {https://arxiv.org/abs/1809.04281}
}

@inproceedings{hawthorne2019enabling,
  title = {Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset},
  author = {Hawthorne, Curtis and Stasyuk, Andriy and Roberts, Adam and Simon, Ian and Huang, Cheng-Zhi Anna and Dieleman, Sander and Elsen, Erich and Engel, Jesse and Eck, Douglas},
  booktitle = {International Conference on Learning Representations},
  year = {2019},
  url = {https://arxiv.org/abs/1810.12247}
}

@inproceedings{huang2020pop,
  title = {Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions},
  author = {Huang, Yu-Siang and Yang, Yi-Hsuan},
  booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
  year = {2020},
  url = {https://arxiv.org/abs/2002.00212}
}

@article{su2021roformer,
  title = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author = {Su, Jianlin and Lu, Yu and Pan, Shengfeng and Murtadha, Ahmed and Wen, Bo and Liu, Yunfeng},
  journal = {arXiv preprint arXiv:2104.09864},
  year = {2021},
  url = {https://arxiv.org/abs/2104.09864}
}

@inproceedings{ainslie2023gqa,
  title = {{GQA}: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
  author = {Ainslie, Joshua and Lee-Thorp, James and de Jong, Michiel and Zemlyanskiy, Yury and Lebron, Federico and Sanghai, Sumit},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages = {4895--4901},
  year = {2023},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2023.emnlp-main.298},
  url = {https://aclanthology.org/2023.emnlp-main.298/}
}

@inproceedings{zhang2019root,
  title = {Root Mean Square Layer Normalization},
  author = {Zhang, Biao and Sennrich, Rico},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {32},
  year = {2019},
  url = {https://papers.neurips.cc/paper/9403-root-mean-square-layer-normalization}
}

@article{shazeer2020glu,
  title = {{GLU} Variants Improve Transformer},
  author = {Shazeer, Noam},
  journal = {arXiv preprint arXiv:2002.05202},
  year = {2020},
  url = {https://arxiv.org/abs/2002.05202}
}

@inproceedings{loshchilov2019decoupled,
  title = {Decoupled Weight Decay Regularization},
  author = {Loshchilov, Ilya and Hutter, Frank},
  booktitle = {International Conference on Learning Representations},
  year = {2019},
  url = {https://arxiv.org/abs/1711.05101}
}

@misc{jordan2024muon,
  title = {Muon: An optimizer for hidden layers in neural networks},
  author = {Keller Jordan and Yuchen Jin and Vlado Boza and Jiacheng You and Franz Cesista and Laker Newhouse and Jeremy Bernstein},
  year = {2024},
  url = {https://kellerjordan.github.io/posts/muon/}
}
```

</details>

## Install Dependencies

```bash
uv sync
```

For better WAV rendering, install FluidSynth: <https://www.fluidsynth.org/wiki/Download/#distributions>

## Training / Eval

Download MAESTRO v3.0.0 into `data/` and tokenize the train/validation splits into REMI sequences:

```bash
uv run python -m src.data --config configs/config.json
```

Train the 17.4M parameter config:

```bash
uv run python -m src.train --config configs/config.json
```

Train the 38.2M parameter config:

```bash
uv run python -m src.train --config configs/38m.json
```

Evaluate a checkpoint on the validation split:

```bash
uv run python -m src.eval \
  --config configs/config.json \
  --checkpoint models/remi-17m/best_model.pt
```

Training writes checkpoints under the `models_dir` in each config. The default config writes to `models/remi-17m/`; the 38.2M config writes to `models/remi-38m/`.

## Model Configs

| config | params | layers | width | heads | kv heads | mlp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `configs/config.json` | 17.4M | 8 | 384 | 6 | 2 | 1536 |
| `configs/38m.json` | 38.2M | 10 | 512 | 8 | 2 | 2048 |

## Inference

Generate a MIDI file from a checkpoint:

```bash
uv run python -m src.generate \
  --config configs/config.json \
  --checkpoint models/remi-17m/best_model.pt \
  --output outputs/sample.mid
```

Render the MIDI to WAV for listening:

```bash
uv run python -m src.render outputs/sample.mid --output outputs/sample.wav
```

`src.render` uses FluidSynth when it is installed. Otherwise it falls back to `pretty_midi` synthesis, which is lower quality.

PyTorch generation uses a KV cache. `src.export` exports a full-sequence ONNX model for standard ONNX Runtime inference. The app also loads cached ONNX Runtime step models named `models/remi-17m/step.onnx` or `models/remi-17m/step-int8.onnx` when those files are present. The CPU int8 path is intended for faster CPU serving and can sound slightly worse than FP32.

If you are running locally and just want to use the app, pull the published model artifacts from the Hugging Face Space:

```bash
uv run --with huggingface_hub python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="anngo-1/piano-ml",
    repo_type="space",
    local_dir=".",
    allow_patterns=[
        "models/remi-17m/best_model.pt",
        "models/remi-17m/step-int8.onnx",
        "models/remi-17m/step.onnx",
        "models/remi-17m/step.onnx.data",
    ],
)
PY
```

Optional audio UI, for listening to generated samples:

```bash
uv sync --extra app
uv run python app.py
```
