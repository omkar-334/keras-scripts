import numpy as np


class DummyLoader:
    def __init__(self, keras_model, hf_model, convert_weights_fn, remove_model_prefix=True):
        self.keras_model = keras_model
        self.hf_model = hf_model
        self.hf_model_weights = hf_model.state_dict()
        self.mismatched_layers = []
        self.num_layers = 0
        self.model_type = hf_model.config.model_type
        self.remove_model_prefix = remove_model_prefix
        self.convert_weights_fn = convert_weights_fn

    def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
        if self.remove_model_prefix:
            hf_weight_key = hf_weight_key.removeprefix(self.model_type + ".")

        print(hf_weight_key)

        status = self.compare_weights(keras_variable, hf_weight_key, hook_fn)
        self.num_layers += 1
        if not status:
            self.mismatched_layers.append(hf_weight_key)
        return status

    def compare_weights(self, keras_variable, hf_weight_key, hook_fn=None, atol=1e-3, rtol=1e-6):
        keras_w = keras_variable
        hf_w = self.hf_model_weights[hf_weight_key]
        if hook_fn:
            hf_w = hook_fn(hf_w, list(keras_w.shape))

        if keras_w.shape != hf_w.shape:
            keras_w = keras_w.transpose(1, 2, 0).reshape(768, 768)
            if keras_w.shape != hf_w.shape:
                print(f"❌ Shape mismatch: Keras {keras_w.shape} vs HF {hf_w.shape}")
                return False

        if not isinstance(keras_w, np.ndarray):
            keras_w = keras_w.numpy()
        if not isinstance(hf_w, np.ndarray):
            hf_w = hf_w.numpy()

        mismatched = np.sum(~np.isclose(keras_w, hf_w, atol=atol, rtol=rtol))
        max_abs_diff = np.max(np.abs(keras_w - hf_w))
        max_rel_diff = np.max(np.abs((keras_w - hf_w) / (hf_w + 1e-10)))  # Avoid division by zero

        output = (
            f"- Mismatched elements: {mismatched} / {keras_w.size} ({mismatched / keras_w.size:.4%})"
            + f" --- Max abs diff: {max_abs_diff:.6f}"
            + f" --- Max rel diff: {max_rel_diff:.6f}"
        )

        print(output)
        return mismatched == 0

    def check_all_weights(loader):
        backbone = loader.keras_model
        print("\n=== Weight Comparison Report ===")

        loader.convert_weights_fn(backbone, loader, transformers_config=None)

        print("\n=== Summary ===")
        if loader.mismatched_layers:
            print(f"❌ {len(loader.mismatched_layers)} / {loader.num_layers} layers have mismatches:")
        else:
            print("✅ All layers match!")

    def print_mismatched_layers(loader):
        if loader.mismatched_layers:
            for layer in loader.mismatched_layers:
                print(f"  - {layer}")


if __name__ == "__main__":
    import keras_hub
    from keras_hub.src.utils.transformers.convert_roberta import convert_weights
    from transformers import BertModel, RobertaModel

    hf_model = RobertaModel.from_pretrained("roberta-base")
    keras_model = keras_hub.models.RobertaBackbone.from_preset("hf://FacebookAI/roberta-base", dtype="bfloat16")

    loader = DummyLoader(keras_model, hf_model, convert_weights)

    loader.check_all_weights()
    # loader.print_mismatched_layers()
