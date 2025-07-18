# Tokenizer Comparison for Named Entity Recognition (NER)

## Description

This project compares how two popular tokenizers — **WordPiece** (`distilbert-base-uncased`) and **Byte-Pair Encoding (BPE)** (`roberta-base`) — affect performance on a named entity recognition (NER) task using the CoNLL-2003 dataset.

## Dataset

- **CoNLL-2003** (via HuggingFace Datasets)
- English NER task with labels: `PER`, `ORG`, `LOC`, `MISC`, `O`

## Sample output

```bash
=== Tokenization Comparison on NER ===
           Tokenizer    f1  accuracy
distilbert-base-uncased 0.89      0.98
          roberta-base  0.91      0.99
