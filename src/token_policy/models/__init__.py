"""Language-conditioned SONIC token policy.

    Text  -> language_backbone (frozen Qwen3-0.6B, cached; outside the model) -> H_L -> skill_program_decoder -> C
    State -> state_encoder -> S
    (C, S) -> motion_transformer -> M -> action_decoder -> A (FSQ logits) -> (k, Z) -> frozen SONIC decoder

config.ModelConfig: hyper-parameters; model.TokenPolicyModel: assembly of the four trainable components
(input H_L, X).  Architecture only: no training, checkpoint I/O, feature extraction or control loop here.
"""
