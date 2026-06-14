//! Model selector with embedded JSON definitions

#[derive(Debug, Clone)]
pub struct Model {
    pub name: &'static str,
    pub family: &'static str,
    pub description: &'static str,
    pub json_content: &'static str,
}

pub static MODEL_LIST: &[Model] = &[
    Model {
        name: "GPT-3 175B",
        family: "Transformer",
        description: "OpenAI's largest GPT-3 model with 175B parameters",
        json_content: include_str!("../../examples/models/gpt3_175b.json"),
    },
    Model {
        name: "LLaMA-2 70B",
        family: "Transformer",
        description: "Meta's LLaMA-2 model with GQA, 70B parameters",
        json_content: include_str!("../../examples/models/llama2_70b.json"),
    },
    Model {
        name: "Mixtral 8x7B",
        family: "MoE",
        description: "Mistral's Mixture of Experts with 8 experts",
        json_content: include_str!("../../examples/models/mixtral_8x7b.json"),
    },
    Model {
        name: "DeepSeek-V3",
        family: "MoE",
        description: "DeepSeek's large MoE model with 256 experts",
        json_content: include_str!("../../examples/models/deepseek_v3.json"),
    },
    Model {
        name: "Mamba 2.8B",
        family: "SSM",
        description: "State Space Model with selective state spaces",
        json_content: include_str!("../../examples/models/mamba_2.8b.json"),
    },
    Model {
        name: "RWKV 7B",
        family: "SSM",
        description: "Receptance Weighted Key Value model",
        json_content: include_str!("../../examples/models/rwkv_7b.json"),
    },
    Model {
        name: "ResNet-50",
        family: "CNN",
        description: "Classic residual network for image classification",
        json_content: include_str!("../../examples/models/resnet50.json"),
    },
    Model {
        name: "VGG-16",
        family: "CNN",
        description: "Oxford's VGG network with 16 layers",
        json_content: include_str!("../../examples/models/vgg16.json"),
    },
    Model {
        name: "Stable Diffusion 1.5",
        family: "Diffusion",
        description: "Text-to-image diffusion model",
        json_content: include_str!("../../examples/models/stable_diffusion_1.5.json"),
    },
    Model {
        name: "SDXL 1.0",
        family: "Diffusion",
        description: "Stable Diffusion XL for high-res generation",
        json_content: include_str!("../../examples/models/sdxl_1.0.json"),
    },
];
