# Implicit Memory Transformer for Computationally Efficient Simultaneous Speech Translation

This repository is a fork of https://github.com/pytorch/fairseq containing the supplementary code used in our ACL 2023 paper Implicit Memory Transformer for Computationally Efficient
Simultaneous Speech Translation.  Our code implementation is in `fairseq/models/speech_to_text/modules/implicit_memory_attention.py`.

If you use this code, please consider citing our paper.

The data preparation script for the MuST-C dataset we used in our paper is `examples/speech_to_text/prep_mustc_data.py`.

The script we used to run the ASR pretraining experiments on a single GPU for the ACL 2023 paper is the following:

```bash

fairseq-train ${data_dir} \
    --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
    --save-dir ${save_dir} --num-workers 2 --max-tokens 80000 --max-update 100000 \
    --task speech_to_text --criterion label_smoothed_cross_entropy \
    --arch implicit_memory_transformer --optimizer adam --adam-betas [0.9,0.98] --lr 0.0007 --lr-scheduler inverse_sqrt \
    --simul-type waitk_fixed_pre_decision --criterion label_smoothed_cross_entropy --fixed-pre-decision-ratio 8 --waitk-lagging 1 \
    --warmup-updates 4000 --warmup-init-lr 0.0001 --clip-norm 10.0 --seed 3 --update-freq 4 \
    --ddp-backend legacy_ddp \
    --log-interval 50 \
    --segment-size 64 --right-context 32 --left-context 32 --max-relative-position 16 --left-context-method pre_output \
    --encoder-normalize-before --decoder-normalize-before --enable-left-grad \
    --patience 5 --keep-last-epochs 5 \
```

In the script, `${data_dir}` refers to the directory of the prepared dataset, and `${save_dir}` refers to the directory to save the model checkpoints.  

Similarly, the script used to run the SimulST pretraining experiments on a single GPU for the ACL 2023 paper is the following:

```bash
fairseq-train ${data_dir} \
    --task speech_to_text --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
    --save-dir ${save_dir} \
    --load-pretrained-encoder-from ${pre_train_dir}/checkpoint_average.pt \
    --arch implicit_memory_transformer \
    --simul-type waitk_fixed_pre_decision --criterion label_smoothed_cross_entropy --fixed-pre-decision-ratio 8 --waitk-lagging 1 \
    --max-tokens 40000 --num-workers 1 --update-freq 8 \
    --optimizer adam --adam-betas [0.9,0.98] --lr 0.00035 --lr-scheduler inverse_sqrt --warmup-updates 7500 --warmup-init-lr 0.0001 --clip-norm 10.0 \
    --max-update 100000 --seed 4 --ddp-backend legacy_ddp --log-interval 50 \
    --segment-size 64 --right-context 32 --left-context 32 --max-relative-position 16 --left-context-method pre_output \
    --encoder-normalize-before --decoder-normalize-before --enable-left-grad \
    --attention-dropout 0.2 --activation-dropout 0.2 --weight-decay 0.0001 \
    --patience 10 --keep-last-epochs 10 \
```

The checkpoint averaging script we used to average model checkpoints after ASR and SimulST training is `scripts/average_checkpoints.py`. 

The data preparation script we used to prepare our test set is `examples/speech_to_text/seg_mustc_data.py`.  

We performed inference on our Implicit Memory Transformer using SimulEval(https://github.com/facebookresearch/SimulEval) version 1.0.1.

# Citation

```bibtex
@inproceedings{raffel-chen-2023-implicit,
    title = "Implicit Memory Transformer for Computationally Efficient Simultaneous Speech Translation",
    author = "Raffel, Matthew  and
      Chen, Lizhong",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.816",
    pages = "12900--12907",
    abstract = "Simultaneous speech translation is an essential communication task difficult for humans whereby a translation is generated concurrently with oncoming speech inputs. For such a streaming task, transformers using block processing to break an input sequence into segments have achieved state-of-the-art performance at a reduced cost. Current methods to allow information to propagate across segments, including left context and memory banks, have faltered as they are both insufficient representations and unnecessarily expensive to compute. In this paper, we propose an Implicit Memory Transformer that implicitly retains memory through a new left context method, removing the need to explicitly represent memory with memory banks. We generate the left context from the attention output of the previous segment and include it in the keys and values of the current segment{'}s attention calculation. Experiments on the MuST-C dataset show that the Implicit Memory Transformer provides a substantial speedup on the encoder forward pass with nearly identical translation quality when compared with the state-of-the-art approach that employs both left context and memory banks.",
}
```

Below, is the original README file.

<p align="center">
  <img src="docs/fairseq_logo.png" width="150">
  <br />
  <br />
  <a href="https://github.com/pytorch/fairseq/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/pytorch/fairseq.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/actions?query=workflow:build"><img alt="Build Status" src="https://github.com/pytorch/fairseq/workflows/build/badge.svg" /></a>
  <a href="https://fairseq.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/fairseq/badge/?version=latest" /></a>
</p>

--------------------------------------------------------------------------------

Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks.

We provide reference implementations of various sequence modeling papers:

<details><summary>List of implemented papers</summary><p>

* **Convolutional Neural Networks (CNN)**
  + [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](examples/language_model/conv_lm/README.md)
  + [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](examples/conv_seq2seq/README.md)
  + [Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018)](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
  + [Hierarchical Neural Story Generation (Fan et al., 2018)](examples/stories/README.md)
  + [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](examples/wav2vec/README.md)
* **LightConv and DynamicConv models**
  + [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](examples/pay_less_attention_paper/README.md)
* **Long Short-Term Memory (LSTM) networks**
  + Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)
* **Transformer (self-attention) networks**
  + Attention Is All You Need (Vaswani et al., 2017)
  + [Scaling Neural Machine Translation (Ott et al., 2018)](examples/scaling_nmt/README.md)
  + [Understanding Back-Translation at Scale (Edunov et al., 2018)](examples/backtranslation/README.md)
  + [Adaptive Input Representations for Neural Language Modeling (Baevski and Auli, 2018)](examples/language_model/README.adaptive_inputs.md)
  + [Lexically constrained decoding with dynamic beam allocation (Post & Vilar, 2018)](examples/constrained_decoding/README.md)
  + [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (Dai et al., 2019)](examples/truncated_bptt/README.md)
  + [Adaptive Attention Span in Transformers (Sukhbaatar et al., 2019)](examples/adaptive_span/README.md)
  + [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](examples/translation_moe/README.md)
  + [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](examples/roberta/README.md)
  + [Facebook FAIR's WMT19 News Translation Task Submission (Ng et al., 2019)](examples/wmt19/README.md)
  + [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](examples/joint_alignment_translation/README.md )
  + [Multilingual Denoising Pre-training for Neural Machine Translation (Liu et at., 2020)](examples/mbart/README.md)
  + [Neural Machine Translation with Byte-Level Subwords (Wang et al., 2020)](examples/byte_level_bpe/README.md)
  + [Unsupervised Quality Estimation for Neural Machine Translation (Fomicheva et al., 2020)](examples/unsupervised_quality_estimation/README.md)
  + [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](examples/wav2vec/README.md)
  + [Generating Medical Reports from Patient-Doctor Conversations Using Sequence-to-Sequence Models (Enarvi et al., 2020)](examples/pointer_generator/README.md)
  + [Linformer: Self-Attention with Linear Complexity (Wang et al., 2020)](examples/linformer/README.md)
  + [Cross-lingual Retrieval for Iterative Self-Supervised Training (Tran et al., 2020)](examples/criss/README.md)
  + [Deep Transformers with Latent Depth (Li et al., 2020)](examples/latent_depth/README.md)
  + [Unsupervised Cross-lingual Representation Learning for Speech Recognition (Conneau et al., 2020)](https://arxiv.org/abs/2006.13979)
  + [Self-training and Pre-training are Complementary for Speech Recognition (Xu et al., 2020)](https://arxiv.org/abs/2010.11430)
  + [Robust wav2vec 2.0: Analyzing Domain Shift in Self-Supervised Pre-Training (Hsu, et al., 2021)](https://arxiv.org/abs/2104.01027)
  + [Unsupervised Speech Recognition (Baevski, et al., 2021)](https://arxiv.org/abs/2105.11084)
  + [Simple and Effective Zero-shot Cross-lingual Phoneme Recognition (Xu et al., 2021)](https://arxiv.org/abs/2109.11680)
  + [VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding (Xu et. al., 2021)](https://arxiv.org/pdf/2109.14084.pdf)
  + [VLM: Task-agnostic Video-Language Model Pre-training for Video Understanding (Xu et. al., 2021)](https://aclanthology.org/2021.findings-acl.370.pdf)
  + [NormFormer: Improved Transformer Pretraining with Extra Normalization (Shleifer et. al, 2021)](examples/normformer/README.md)
* **Non-autoregressive Transformers**
  + Non-Autoregressive Neural Machine Translation (Gu et al., 2017)
  + Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement (Lee et al. 2018)
  + Insertion Transformer: Flexible Sequence Generation via Insertion Operations (Stern et al. 2019)
  + Mask-Predict: Parallel Decoding of Conditional Masked Language Models (Ghazvininejad et al., 2019)
  + [Levenshtein Transformer (Gu et al., 2019)](examples/nonautoregressive_translation/README.md)
* **Finetuning**
  + [Better Fine-Tuning by Reducing Representational Collapse (Aghajanyan et al. 2020)](examples/rxf/README.md)

</p></details>

### What's New:
* December 2021 [Released Direct speech-to-speech translation code](examples/speech_to_speech/README.md)
* October 2021 [Released VideoCLIP and VLM models](examples/MMPT/README.md)
* October 2021 [Released multilingual finetuned XLSR-53 model](examples/wav2vec/README.md)
* September 2021 [`master` branch renamed to `main`](https://github.com/github/renaming).
* July 2021 [Released DrNMT code](examples/discriminative_reranking_nmt/README.md)
* July 2021 [Released Robust wav2vec 2.0 model](examples/wav2vec/README.md)
* June 2021 [Released XLMR-XL and XLMR-XXL models](examples/xlmr/README.md)
* May 2021 [Released Unsupervised Speech Recognition code](examples/wav2vec/unsupervised/README.md)
* March 2021 [Added full parameter and optimizer state sharding + CPU offloading](examples/fully_sharded_data_parallel/README.md)
* February 2021 [Added LASER training code](examples/laser/README.md)
* December 2020: [Added Adaptive Attention Span code](examples/adaptive_span/README.md)
* December 2020: [GottBERT model and code released](examples/gottbert/README.md)
* November 2020: Adopted the [Hydra](https://github.com/facebookresearch/hydra) configuration framework
  * [see documentation explaining how to use it for new and existing projects](docs/hydra_integration.md)
* November 2020: [fairseq 0.10.0 released](https://github.com/pytorch/fairseq/releases/tag/v0.10.0)
* October 2020: [Added R3F/R4F (Better Fine-Tuning) code](examples/rxf/README.md)
* October 2020: [Deep Transformer with Latent Depth code released](examples/latent_depth/README.md)
* October 2020: [Added CRISS models and code](examples/criss/README.md)

<details><summary>Previous updates</summary><p>

* September 2020: [Added Linformer code](examples/linformer/README.md)
* September 2020: [Added pointer-generator networks](examples/pointer_generator/README.md)
* August 2020: [Added lexically constrained decoding](examples/constrained_decoding/README.md)
* August 2020: [wav2vec2 models and code released](examples/wav2vec/README.md)
* July 2020: [Unsupervised Quality Estimation code released](examples/unsupervised_quality_estimation/README.md)
* May 2020: [Follow fairseq on Twitter](https://twitter.com/fairseq)
* April 2020: [Monotonic Multihead Attention code released](examples/simultaneous_translation/README.md)
* April 2020: [Quant-Noise code released](examples/quant_noise/README.md)
* April 2020: [Initial model parallel support and 11B parameters unidirectional LM released](examples/megatron_11b/README.md)
* March 2020: [Byte-level BPE code released](examples/byte_level_bpe/README.md)
* February 2020: [mBART model and code released](examples/mbart/README.md)
* February 2020: [Added tutorial for back-translation](https://github.com/pytorch/fairseq/tree/main/examples/backtranslation#training-your-own-model-wmt18-english-german)
* December 2019: [fairseq 0.9.0 released](https://github.com/pytorch/fairseq/releases/tag/v0.9.0)
* November 2019: [VizSeq released (a visual analysis toolkit for evaluating fairseq models)](https://facebookresearch.github.io/vizseq/docs/getting_started/fairseq_example)
* November 2019: [CamemBERT model and code released](examples/camembert/README.md)
* November 2019: [BART model and code released](examples/bart/README.md)
* November 2019: [XLM-R models and code released](examples/xlmr/README.md)
* September 2019: [Nonautoregressive translation code released](examples/nonautoregressive_translation/README.md)
* August 2019: [WMT'19 models released](examples/wmt19/README.md)
* July 2019: fairseq relicensed under MIT license
* July 2019: [RoBERTa models and code released](examples/roberta/README.md)
* June 2019: [wav2vec models and code released](examples/wav2vec/README.md)

</p></details>

### Features:

* multi-GPU training on one machine or across multiple machines (data and model parallel)
* fast generation on both CPU and GPU with multiple search algorithms implemented:
  + beam search
  + Diverse Beam Search ([Vijayakumar et al., 2016](https://arxiv.org/abs/1610.02424))
  + sampling (unconstrained, top-k and top-p/nucleus)
  + [lexically constrained decoding](examples/constrained_decoding/README.md) (Post & Vilar, 2018)
* [gradient accumulation](https://fairseq.readthedocs.io/en/latest/getting_started.html#large-mini-batch-training-with-delayed-updates) enables training with large mini-batches even on a single GPU
* [mixed precision training](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-with-half-precision-floating-point-fp16) (trains faster with less GPU memory on [NVIDIA tensor cores](https://developer.nvidia.com/tensor-cores))
* [extensible](https://fairseq.readthedocs.io/en/latest/overview.html): easily register new models, criterions, tasks, optimizers and learning rate schedulers
* [flexible configuration](docs/hydra_integration.md) based on [Hydra](https://github.com/facebookresearch/hydra) allowing a combination of code, command-line and file based configuration
* [full parameter and optimizer state sharding](examples/fully_sharded_data_parallel/README.md)
* [offloading parameters to CPU](examples/fully_sharded_data_parallel/README.md)

We also provide [pre-trained models for translation and language modeling](#pre-trained-models-and-examples)
with a convenient `torch.hub` interface:

``` python
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')
en2de.translate('Hello world', beam=5)
# 'Hallo Welt'
```

See the PyTorch Hub tutorials for [translation](https://pytorch.org/hub/pytorch_fairseq_translation/)
and [RoBERTa](https://pytorch.org/hub/pytorch_fairseq_roberta/) for more examples.

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# to install the latest stable release (0.10.x)
# pip install fairseq
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size`
 as command line options to `nvidia-docker run` .

# Getting Started

The [full documentation](https://fairseq.readthedocs.io/) contains instructions
for getting started, training new models and extending fairseq with new model
types and tasks.

# Pre-trained models and examples

We provide pre-trained models and pre-processed, binarized test sets for several tasks listed below,
as well as example training and evaluation commands.

* [Translation](examples/translation/README.md): convolutional and transformer models are available
* [Language Modeling](examples/language_model/README.md): convolutional and transformer models are available

We also have more detailed READMEs to reproduce results from specific papers:

* [XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale (Babu et al., 2021)](examples/wav2vec/xlsr/README.md)
* [Cross-lingual Retrieval for Iterative Self-Supervised Training (Tran et al., 2020)](examples/criss/README.md)
* [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](examples/wav2vec/README.md)
* [Unsupervised Quality Estimation for Neural Machine Translation (Fomicheva et al., 2020)](examples/unsupervised_quality_estimation/README.md)
* [Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)](examples/quant_noise/README.md)
* [Neural Machine Translation with Byte-Level Subwords (Wang et al., 2020)](examples/byte_level_bpe/README.md)
* [Multilingual Denoising Pre-training for Neural Machine Translation (Liu et at., 2020)](examples/mbart/README.md)
* [Reducing Transformer Depth on Demand with Structured Dropout (Fan et al., 2019)](examples/layerdrop/README.md)
* [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](examples/joint_alignment_translation/README.md)
* [Levenshtein Transformer (Gu et al., 2019)](examples/nonautoregressive_translation/README.md)
* [Facebook FAIR's WMT19 News Translation Task Submission (Ng et al., 2019)](examples/wmt19/README.md)
* [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](examples/roberta/README.md)
* [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](examples/wav2vec/README.md)
* [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](examples/translation_moe/README.md)
* [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](examples/pay_less_attention_paper/README.md)
* [Understanding Back-Translation at Scale (Edunov et al., 2018)](examples/backtranslation/README.md)
* [Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018)](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
* [Hierarchical Neural Story Generation (Fan et al., 2018)](examples/stories/README.md)
* [Scaling Neural Machine Translation (Ott et al., 2018)](examples/scaling_nmt/README.md)
* [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](examples/conv_seq2seq/README.md)
* [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](examples/language_model/README.conv.md)

# Join the fairseq community

* Twitter: https://twitter.com/fairseq
* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users

# License

fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
