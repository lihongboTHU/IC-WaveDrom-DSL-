import paddle
from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.generation import GenerationConfig

model_path = "./IC-WaveDrom-DSL" 
image = Image.open("./dataset/images/blur_001.png").convert("RGB") # 测试一张图片,请替换为你自己的图片路径

model = AutoModelForConditionalGeneration.from_pretrained( # 自动读取模型文件
    model_path, 
    convert_from_hf=True
).eval()

model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

processor = AutoProcessor.from_pretrained(model_path)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "WaveDrom Recognition:"},
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, 
    return_dict=True, return_tensors="pd"
)

generation_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=1024,
    pad_token_id=0,
    eos_token_id=2
)

with paddle.no_grad():
    outputs = model.generate(**inputs, generation_config=generation_config)
    output_text = processor.decode(outputs[0].tolist()[0], skip_special_tokens=True)

print(output_text)