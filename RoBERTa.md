| RobertaBackbone                                    | HF RobertaModel                                |
| -------------------------------------------------- | ---------------------------------------------- |
| ==self.embeddings.token_embedding==                | embeddings.word_embeddings                     |
| ==self.embeddings.position_embedding==             | embeddings.position_embeddings                 |
|                                                    | embeddings.token_type_embeddings               |
| ==self.embeddings_layer_norm==                     | embeddings.LayerNorm                           |
| ==~~self.embeddings_dropout~~==                    | embeddings.dropout                             |
| MultiHeadAttention.query_dense                     | encoder.layer.attention.self.query             |
| MultiHeadAttention.key_dense                       | encoder.layer.attention.self.key               |
| MultiHeadAttention.value_dense                     | encoder.layer.attention.self.value             |
| ~~MultiHeadAttention._dropout~~                    | encoder.layer.attention.self.dropout           |
| MultiHeadAttention.output_dense                    | encoder.layer.attention.output.dense           |
| TransformerEncoder._self_attention_layer_norm      | encoder.layer.attention.output.LayerNorm       |
| ~~TransformerEncoder._self_attention_dropout~~     | encoder.layer.attention.output.dropout         |
| TransformerEncoder._feedforward_intermediate_dense | encoder.layer.intermediate.dense               |
|                                                    | encoder.layer.intermediate.intermediate_act_fn |
| TransformerEncoder._feedforward_output_dense       | encoder.layer.output.dense                     |
| TransformerEncoder._feedforward_layer_norm         | encoder.layer.output.LayerNorm                 |
| ~~TransformerEncoder._feedforward_dropout~~        | encoder.layer.output.dropout                   |
|                                                    | pooler.dense                                   |
|                                                    | pooler.activation                              |
# RobertaModel
```
RobertaModel(
  (embeddings): RobertaEmbeddings(
    (word_embeddings): Embedding(50265, 768, padding_idx=1)
    (position_embeddings): Embedding(514, 768, padding_idx=1)
    (token_type_embeddings): Embedding(1, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): RobertaEncoder(
    (layer): ModuleList(
      (0-11): 12 x RobertaLayer(
        (attention): RobertaAttention(
          (self): RobertaSdpaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): RobertaSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): RobertaIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): RobertaOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (pooler): RobertaPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)
```

# RobertaConfig
```
RobertaConfig {
  "_attn_implementation_autoset": true,
  "_name_or_path": "roberta-base",
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.48.3",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 50265
}
```
