#!/usr/bin/env python3
"""Generate 20 test models for neurax compiler testing."""
import json, os

models = [
    # Already created: 01, 02, 03
    # 04 - Mixtral MoE 8x7B
    {"file": "04_mixtral_8x7b.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "Mixtral-8x7B", "type": "moe", "layers": [
            {"id": "embed", "layer_type": "embedding", "params": {"vocab_size": 32000, "hidden_size": 4096}},
            *[{"id": f"layer_{i}", "layer_type": "moe", "params": {"hidden_size": 4096, "num_attention_heads": 32, "intermediate_size": 14336, "num_experts": 8, "top_k": 2}} for i in range(8)],
            {"id": "ln_f", "layer_type": "normalization", "params": {"hidden_size": 4096}}
        ]},
        "training": {"batch_size": 16, "sequence_length": 4096, "precision": "bf16", "learning_rate": 0.0001, "num_epochs": 1},
        "hardware": {"gpus": [{"name": "H100", "memory_gb": 80, "count": 32}]},
        "data": {"dataset_size": 1000000000000, "vocab_size": 32000, "num_classes": 0}
    }},
    # 05 - ViT-Large
    {"file": "05_vit_large.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "ViT-Large-307M", "type": "transformer", "layers": [
            {"id": "patch_embed", "layer_type": "conv", "params": {"in_channels": 3, "out_channels": 1024, "kernel_size": 16, "hidden_size": 1024}},
            *[{"id": f"layer_{i}", "layer_type": "attention", "params": {"hidden_size": 1024, "num_attention_heads": 16, "intermediate_size": 4096}} for i in range(24)],
            {"id": "ln_f", "layer_type": "normalization", "params": {"hidden_size": 1024}},
            {"id": "classifier", "layer_type": "linear", "params": {"hidden_size": 1024, "output_size": 1000}}
        ]},
        "training": {"batch_size": 256, "sequence_length": 197, "precision": "fp16", "learning_rate": 0.001, "num_epochs": 300},
        "hardware": {"gpus": [{"name": "A100", "memory_gb": 80, "count": 8}]},
        "data": {"dataset_size": 14000000, "vocab_size": 0, "num_classes": 1000}
    }},
    # 06 - ResNet-50 CNN
    {"file": "06_resnet50.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "ResNet-50-25M", "type": "transformer", "layers": [
            {"id": "conv1", "layer_type": "conv", "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 7, "hidden_size": 64}},
            {"id": "block1", "layer_type": "attention", "params": {"hidden_size": 256, "num_attention_heads": 4}},
            {"id": "block2", "layer_type": "attention", "params": {"hidden_size": 512, "num_attention_heads": 8}},
            {"id": "block3", "layer_type": "attention", "params": {"hidden_size": 1024, "num_attention_heads": 16}},
            {"id": "block4", "layer_type": "attention", "params": {"hidden_size": 2048, "num_attention_heads": 32}},
            {"id": "classifier", "layer_type": "linear", "params": {"hidden_size": 2048, "output_size": 1000}}
        ]},
        "training": {"batch_size": 256, "sequence_length": 196, "precision": "fp32", "learning_rate": 0.1, "num_epochs": 90},
        "hardware": {"gpus": [{"name": "V100", "memory_gb": 32, "count": 8}]},
        "data": {"dataset_size": 1281167, "vocab_size": 0, "num_classes": 1000}
    }},
    # 07 - Mamba-3B SSM
    {"file": "07_mamba_3b.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "Mamba-3B", "type": "ssm", "layers": [
            {"id": "embed", "layer_type": "embedding", "params": {"vocab_size": 50280, "hidden_size": 2560}},
            *[{"id": f"mamba_{i}", "layer_type": "mamba", "params": {"d_model": 2560, "d_state": 16, "d_conv": 4, "expand": 2},
               "input_shape": [2048, 2560], "output_shape": [2048, 2560]} for i in range(32)],
            {"id": "ln_f", "layer_type": "normalization", "params": {"hidden_size": 2560}},
            {"id": "lm_head", "layer_type": "linear", "params": {"in_features": 2560, "out_features": 50280, "bias": False},
             "input_shape": [2048, 2560], "output_shape": [2048, 50280]}
        ]},
        "training": {"batch_size": 8, "sequence_length": 2048, "precision": "bf16", "learning_rate": 0.0006, "num_epochs": 1},
        "hardware": {"gpus": [{"name": "A100", "memory_gb": 80, "count": 4}]},
        "data": {"dataset_size": 300000000000, "vocab_size": 50280, "num_classes": 0}
    }},
    # 08 - Stable Diffusion UNet
    {"file": "08_sd_unet.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "StableDiffusion-UNet-860M", "type": "diffusion", "layers": [
            {"id": "conv_in", "layer_type": "conv", "params": {"in_channels": 4, "out_channels": 320, "kernel_size": 3, "hidden_size": 320}},
            {"id": "down_0", "layer_type": "attention", "params": {"hidden_size": 320, "num_attention_heads": 8}},
            {"id": "down_1", "layer_type": "attention", "params": {"hidden_size": 640, "num_attention_heads": 10}},
            {"id": "down_2", "layer_type": "attention", "params": {"hidden_size": 1280, "num_attention_heads": 20}},
            {"id": "mid", "layer_type": "attention", "params": {"hidden_size": 1280, "num_attention_heads": 20}},
            {"id": "up_0", "layer_type": "attention", "params": {"hidden_size": 1280, "num_attention_heads": 20}},
            {"id": "up_1", "layer_type": "attention", "params": {"hidden_size": 640, "num_attention_heads": 10}},
            {"id": "up_2", "layer_type": "attention", "params": {"hidden_size": 320, "num_attention_heads": 8}},
            {"id": "conv_out", "layer_type": "conv", "params": {"in_channels": 320, "out_channels": 4, "kernel_size": 3, "hidden_size": 320}}
        ]},
        "training": {"batch_size": 4, "sequence_length": 4096, "precision": "fp16", "learning_rate": 0.0001, "num_epochs": 200},
        "hardware": {"gpus": [{"name": "A100", "memory_gb": 80, "count": 8}]},
        "data": {"dataset_size": 2000000000, "vocab_size": 49408, "num_classes": 0}
    }},
    # 09 - LSTM Seq2Seq
    {"file": "09_bilstm_seq2seq.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "BiLSTM-Seq2Seq-50M", "type": "transformer", "layers": [
            {"id": "embed_enc", "layer_type": "embedding", "params": {"vocab_size": 32000, "hidden_size": 512}},
            {"id": "lstm_enc_0", "layer_type": "attention", "params": {"hidden_size": 512, "num_attention_heads": 8}},
            {"id": "lstm_enc_1", "layer_type": "attention", "params": {"hidden_size": 512, "num_attention_heads": 8}},
            {"id": "embed_dec", "layer_type": "embedding", "params": {"vocab_size": 32000, "hidden_size": 512}},
            {"id": "lstm_dec_0", "layer_type": "attention", "params": {"hidden_size": 512, "num_attention_heads": 8}},
            {"id": "lstm_dec_1", "layer_type": "attention", "params": {"hidden_size": 512, "num_attention_heads": 8}},
            {"id": "output_proj", "layer_type": "linear", "params": {"hidden_size": 512, "output_size": 32000}}
        ]},
        "training": {"batch_size": 64, "sequence_length": 256, "precision": "fp32", "learning_rate": 0.001, "num_epochs": 30},
        "hardware": {"gpus": [{"name": "V100", "memory_gb": 32, "count": 1}]},
        "data": {"dataset_size": 5000000, "vocab_size": 32000, "num_classes": 0}
    }},
    # 10 - DeepSeek-V3 MoE 671B
    {"file": "10_deepseek_v3.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "DeepSeek-V3-671B", "type": "moe", "layers": [
            {"id": "embed", "layer_type": "embedding", "params": {"vocab_size": 102400, "hidden_size": 7168}},
            *[{"id": f"layer_{i}", "layer_type": "moe", "params": {"hidden_size": 7168, "num_attention_heads": 128, "intermediate_size": 18432, "num_experts": 256, "top_k": 8}} for i in range(6)],
            {"id": "ln_f", "layer_type": "normalization", "params": {"hidden_size": 7168}}
        ]},
        "training": {"batch_size": 8, "sequence_length": 4096, "precision": "bf16", "learning_rate": 0.000022, "num_epochs": 1},
        "hardware": {"gpus": [{"name": "H100", "memory_gb": 80, "count": 2048}]},
        "data": {"dataset_size": 14700000000000, "vocab_size": 102400, "num_classes": 0}
    }},
    # 11 - T5-Base Encoder-Decoder
    {"file": "11_t5_base.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "T5-Base-220M", "type": "transformer", "layers": [
            {"id": "embed", "layer_type": "embedding", "params": {"vocab_size": 32128, "hidden_size": 768}},
            *[{"id": f"enc_{i}", "layer_type": "attention", "params": {"hidden_size": 768, "num_attention_heads": 12, "intermediate_size": 2048}} for i in range(12)],
            *[{"id": f"dec_{i}", "layer_type": "attention", "params": {"hidden_size": 768, "num_attention_heads": 12, "intermediate_size": 2048}} for i in range(12)],
            {"id": "ln_f", "layer_type": "normalization", "params": {"hidden_size": 768}}
        ]},
        "training": {"batch_size": 128, "sequence_length": 512, "precision": "fp32", "learning_rate": 0.001, "num_epochs": 10},
        "hardware": {"gpus": [{"name": "A100", "memory_gb": 80, "count": 4}]},
        "data": {"dataset_size": 750000000000, "vocab_size": 32128, "num_classes": 0}
    }},
    # 12 - Whisper-Large (Audio)
    {"file": "12_whisper_large.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "Whisper-Large-V3-1.5B", "type": "transformer", "layers": [
            {"id": "conv1", "layer_type": "conv", "params": {"in_channels": 128, "out_channels": 1280, "kernel_size": 3, "hidden_size": 1280}},
            {"id": "conv2", "layer_type": "conv", "params": {"in_channels": 1280, "out_channels": 1280, "kernel_size": 3, "hidden_size": 1280}},
            *[{"id": f"enc_{i}", "layer_type": "attention", "params": {"hidden_size": 1280, "num_attention_heads": 20, "intermediate_size": 5120}} for i in range(32)],
            *[{"id": f"dec_{i}", "layer_type": "attention", "params": {"hidden_size": 1280, "num_attention_heads": 20, "intermediate_size": 5120}} for i in range(32)],
            {"id": "ln_f", "layer_type": "normalization", "params": {"hidden_size": 1280}}
        ]},
        "training": {"batch_size": 16, "sequence_length": 1500, "precision": "fp16", "learning_rate": 0.0001, "num_epochs": 3},
        "hardware": {"gpus": [{"name": "A100", "memory_gb": 80, "count": 32}]},
        "data": {"dataset_size": 5000000, "vocab_size": 51866, "num_classes": 0}
    }},
    # 13 - EfficientNet-B7 CNN
    {"file": "13_efficientnet_b7.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "EfficientNet-B7-66M", "type": "transformer", "layers": [
            {"id": "stem", "layer_type": "conv", "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "hidden_size": 64}},
            {"id": "block1", "layer_type": "attention", "params": {"hidden_size": 32, "num_attention_heads": 4}},
            {"id": "block2", "layer_type": "attention", "params": {"hidden_size": 48, "num_attention_heads": 4}},
            {"id": "block3", "layer_type": "attention", "params": {"hidden_size": 80, "num_attention_heads": 8}},
            {"id": "block4", "layer_type": "attention", "params": {"hidden_size": 160, "num_attention_heads": 8}},
            {"id": "block5", "layer_type": "attention", "params": {"hidden_size": 224, "num_attention_heads": 8}},
            {"id": "block6", "layer_type": "attention", "params": {"hidden_size": 384, "num_attention_heads": 16}},
            {"id": "block7", "layer_type": "attention", "params": {"hidden_size": 640, "num_attention_heads": 16}},
            {"id": "classifier", "layer_type": "linear", "params": {"hidden_size": 640, "output_size": 1000}}
        ]},
        "training": {"batch_size": 32, "sequence_length": 600, "precision": "fp32", "learning_rate": 0.016, "num_epochs": 350},
        "hardware": {"gpus": [{"name": "V100", "memory_gb": 32, "count": 4}]},
        "data": {"dataset_size": 1281167, "vocab_size": 0, "num_classes": 1000}
    }},
    # 14 - GPT-J 6B
    {"file": "14_gptj_6b.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "GPT-J-6B", "type": "transformer", "layers": [
            {"id": "embed", "layer_type": "embedding", "params": {"vocab_size": 50400, "hidden_size": 4096}},
            *[{"id": f"layer_{i}", "layer_type": "attention", "params": {"hidden_size": 4096, "num_attention_heads": 16, "intermediate_size": 16384}} for i in range(28)],
            {"id": "ln_f", "layer_type": "normalization", "params": {"hidden_size": 4096}}
        ]},
        "training": {"batch_size": 8, "sequence_length": 2048, "precision": "fp16", "learning_rate": 0.00012, "num_epochs": 1},
        "hardware": {"gpus": [{"name": "A100", "memory_gb": 80, "count": 8}]},
        "data": {"dataset_size": 402000000000, "vocab_size": 50400, "num_classes": 0}
    }},
    # 15 - Falcon-40B
    {"file": "15_falcon_40b.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "Falcon-40B", "type": "transformer", "layers": [
            {"id": "embed", "layer_type": "embedding", "params": {"vocab_size": 65024, "hidden_size": 8192}},
            *[{"id": f"layer_{i}", "layer_type": "attention", "params": {"hidden_size": 8192, "num_attention_heads": 64, "intermediate_size": 32768}} for i in range(10)],
            {"id": "ln_f", "layer_type": "normalization", "params": {"hidden_size": 8192}}
        ]},
        "training": {"batch_size": 16, "sequence_length": 2048, "precision": "bf16", "learning_rate": 0.00006, "num_epochs": 1},
        "hardware": {"gpus": [{"name": "A100", "memory_gb": 80, "count": 64}]},
        "data": {"dataset_size": 1000000000000, "vocab_size": 65024, "num_classes": 0}
    }},
    # 16 - CLIP ViT-L/14 (Multimodal)
    {"file": "16_clip_vit_l14.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "CLIP-ViT-L14-427M", "type": "transformer", "layers": [
            {"id": "patch_embed", "layer_type": "conv", "params": {"in_channels": 3, "out_channels": 1024, "kernel_size": 14, "hidden_size": 1024}},
            *[{"id": f"vis_{i}", "layer_type": "attention", "params": {"hidden_size": 1024, "num_attention_heads": 16, "intermediate_size": 4096}} for i in range(24)],
            {"id": "text_embed", "layer_type": "embedding", "params": {"vocab_size": 49408, "hidden_size": 768}},
            *[{"id": f"txt_{i}", "layer_type": "attention", "params": {"hidden_size": 768, "num_attention_heads": 12, "intermediate_size": 3072}} for i in range(12)],
            {"id": "ln_f", "layer_type": "normalization", "params": {"hidden_size": 768}}
        ]},
        "training": {"batch_size": 32768, "sequence_length": 77, "precision": "fp16", "learning_rate": 0.0005, "num_epochs": 32},
        "hardware": {"gpus": [{"name": "A100", "memory_gb": 80, "count": 256}]},
        "data": {"dataset_size": 400000000, "vocab_size": 49408, "num_classes": 0}
    }},
    # 17 - Phi-3 Mini 3.8B
    {"file": "17_phi3_mini.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "Phi-3-Mini-3.8B", "type": "transformer", "layers": [
            {"id": "embed", "layer_type": "embedding", "params": {"vocab_size": 32064, "hidden_size": 3072}},
            *[{"id": f"layer_{i}", "layer_type": "attention", "params": {"hidden_size": 3072, "num_attention_heads": 32, "intermediate_size": 8192}} for i in range(32)],
            {"id": "ln_f", "layer_type": "normalization", "params": {"hidden_size": 3072}}
        ]},
        "training": {"batch_size": 16, "sequence_length": 4096, "precision": "bf16", "learning_rate": 0.0001, "num_epochs": 1},
        "hardware": {"gpus": [{"name": "A100", "memory_gb": 80, "count": 8}]},
        "data": {"dataset_size": 3300000000000, "vocab_size": 32064, "num_classes": 0}
    }},
    # 18 - Gemma-2B
    {"file": "18_gemma_2b.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "Gemma-2B", "type": "transformer", "layers": [
            {"id": "embed", "layer_type": "embedding", "params": {"vocab_size": 256128, "hidden_size": 2048}},
            *[{"id": f"layer_{i}", "layer_type": "attention", "params": {"hidden_size": 2048, "num_attention_heads": 8, "intermediate_size": 16384}} for i in range(18)],
            {"id": "ln_f", "layer_type": "normalization", "params": {"hidden_size": 2048}}
        ]},
        "training": {"batch_size": 16, "sequence_length": 8192, "precision": "bf16", "learning_rate": 0.0001, "num_epochs": 1},
        "hardware": {"gpus": [{"name": "A100", "memory_gb": 80, "count": 16}]},
        "data": {"dataset_size": 2000000000000, "vocab_size": 256128, "num_classes": 0}
    }},
    # 19 - DINOv2-Giant (Vision SSL)
    {"file": "19_dinov2_giant.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "DINOv2-Giant-1.1B", "type": "transformer", "layers": [
            {"id": "patch_embed", "layer_type": "conv", "params": {"in_channels": 3, "out_channels": 1536, "kernel_size": 14, "hidden_size": 1536}},
            *[{"id": f"layer_{i}", "layer_type": "attention", "params": {"hidden_size": 1536, "num_attention_heads": 24, "intermediate_size": 6144}} for i in range(40)],
            {"id": "ln_f", "layer_type": "normalization", "params": {"hidden_size": 1536}},
            {"id": "head", "layer_type": "linear", "params": {"hidden_size": 1536, "output_size": 65536}}
        ]},
        "training": {"batch_size": 1024, "sequence_length": 257, "precision": "fp16", "learning_rate": 0.002, "num_epochs": 500},
        "hardware": {"gpus": [{"name": "A100", "memory_gb": 80, "count": 16}]},
        "data": {"dataset_size": 142000000, "vocab_size": 0, "num_classes": 0}
    }},
    # 20 - Qwen2-72B (Large LLM)
    {"file": "20_qwen2_72b.json", "data": {
        "schema_version": "1.0.0",
        "model": {"name": "Qwen2-72B", "type": "transformer", "layers": [
            {"id": "embed", "layer_type": "embedding", "params": {"vocab_size": 152064, "hidden_size": 8192}},
            *[{"id": f"layer_{i}", "layer_type": "attention", "params": {"hidden_size": 8192, "num_attention_heads": 64, "intermediate_size": 29568}} for i in range(10)],
            {"id": "ln_f", "layer_type": "normalization", "params": {"hidden_size": 8192}}
        ]},
        "training": {"batch_size": 8, "sequence_length": 32768, "precision": "bf16", "learning_rate": 0.00003, "num_epochs": 1},
        "hardware": {"gpus": [{"name": "H100", "memory_gb": 80, "count": 256}]},
        "data": {"dataset_size": 7000000000000, "vocab_size": 152064, "num_classes": 0}
    }},
]

outdir = os.path.dirname(os.path.abspath(__file__))
for m in models:
    path = os.path.join(outdir, m["file"])
    with open(path, "w") as f:
        json.dump(m["data"], f, indent=2)
    print(f"  Created {m['file']}")

print(f"\nGenerated {len(models)} models in {outdir}")
