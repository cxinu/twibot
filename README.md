# AdaWongBot

Heterophily-aware Graph Transformer (HGT) for Twitter bot detection.

more details in [WRITEUP.md](WRITEUP.md)

TwiBot-20 Dataset is request access only due to privacy reasons as described in https://github.com/BunsenFeng/TwiBot-20,
dataset link https://drive.google.com/drive/folders/1YY_fFK-ZS0lFDvoLaijoCp0yQ1CnG33i

once you get the access download `Twibot-20_processed_data.zip`, because it has roBERTa embeddings of tweets required for AdaRelBot, otherwise you'll have to preprocess and generate these embeddings yourself, (~30min on RTX3060)

also recommended GPU with at least ~4 GB VRAM


## Cite

## reference

- https://distill.pub/2021/gnn-intro/
- https://distill.pub/2021/understanding-gnns/
- Twibot-20 Dataset paper: https://arxiv.org/pdf/2106.13088
- RGCN paper: https://arxiv.org/pdf/1703.06103
- BotRGCN paper: https://arxiv.org/pdf/2106.13092
- RoBERTa paper: https://arxiv.org/pdf/1907.11692
- RGT paper: https://arxiv.org/pdf/2109.02927

## future work refs

- BotUmc paper (SOTA, LLM based): https://arxiv.org/pdf/2503.03775
- BotDGT paper: https://arxiv.org/pdf/2404.15070
