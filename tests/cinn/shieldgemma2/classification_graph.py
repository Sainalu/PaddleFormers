"""Tensor-only boundaries for image-policy binary classification."""

import paddle
from paddle.nn import functional as functional


class ShieldGemma2InferGraph(paddle.nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.final_logit_softcapping = model.config.text_config.final_logit_softcapping

    def forward(self, input_ids, pixel_values, attention_mask, position_ids, token_type_ids):
        outputs = self.model.model(
            input_ids,
            pixel_values,
            attention_mask,
            None,
            position_ids,
            None,
            token_type_ids,
            None,
            None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        hidden_states = outputs[0][:, -1:, :]
        token_indices = paddle.to_tensor([self.model.yes_token_index, self.model.no_token_index], dtype="int64")
        selected_weight = paddle.index_select(self.model.lm_head.weight, token_indices, axis=0)
        logits = paddle.matmul(hidden_states, selected_weight, transpose_y=True)
        if self.model.lm_head.bias is not None:
            logits = logits + paddle.index_select(self.model.lm_head.bias, token_indices, axis=0)
        if self.final_logit_softcapping is not None:
            logits = paddle.tanh(logits / self.final_logit_softcapping) * self.final_logit_softcapping
        logits = logits[:, 0, :].astype("float32")
        return logits, functional.softmax(logits, axis=-1)


class ShieldGemma2TrainGraph(paddle.nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.classifier = ShieldGemma2InferGraph(model)

    def forward(self, input_ids, pixel_values, attention_mask, position_ids, token_type_ids, labels):
        logits, probabilities = self.classifier(
            input_ids, pixel_values, attention_mask, position_ids, token_type_ids
        )
        return functional.cross_entropy(logits, labels)
