import gradio as gr
import paddle
from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.generation import GenerationConfig
import json
import re

# 加载模型逻辑与 demo.py 保持一致
model_path = "./IC-WaveDrom-DSL"

print("正在加载模型，请稍候...")
try:
    model = AutoModelForConditionalGeneration.from_pretrained(
        model_path, 
        convert_from_hf=True
    ).eval()

    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"

    processor = AutoProcessor.from_pretrained(model_path)
    print("模型加载完成！")
except Exception as e:
    print(f"模型加载失败，请确保 {model_path} 路径正确并包含所需文件。错误信息: {e}")
    # 占位符，避免 gradio 启动失败
    model = None
    processor = None

generation_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=1024,
    pad_token_id=0,
    eos_token_id=2
)

def process_image(image_pil):
    if model is None or processor is None:
        return "模型未正确加载，请检查后台日志。", "模型加载失败"
        
    image = image_pil.convert("RGB")
    
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

    with paddle.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)
        output_text = processor.decode(outputs[0].tolist()[0], skip_special_tokens=True)
    
    # 尝试提取 JSON 部分
    json_str = output_text
    match = re.search(r"```json\n(.*?)\n```", output_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        
    # 为了让 WaveDrom 渲染，我们需要构建 HTML 字符串嵌入
    html_content = f"""
    <div style="background-color: white; padding: 20px; border-radius: 8px;">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/wavedrom/3.1.0/skins/default.js" type="text/javascript"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/wavedrom/3.1.0/wavedrom.min.js" type="text/javascript"></script>
        <body onload="WaveDrom.ProcessAll()">
            <script type="WaveDrom">
            {json_str}
            </script>
        </body>
    </div>
    """
    
    return output_text, html_content

# 构建 Gradio 界面
with gr.Blocks(title="IC WaveDrom DSL 转译系统") as demo:
    gr.Markdown("# 🚀 IC 芯片时序图逆向转译系统 Demo")
    gr.Markdown("上传一张 IC 芯片时序波形图，基于 PaddleOCR-VL 微调的多模态大模型将自动解析并生成 WaveDrom DSL 代码，同时为您渲染出波形图像。")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="上传时序波形图")
            submit_btn = gr.Button("开始解析", variant="primary")
            
        with gr.Column(scale=1):
            output_code = gr.Code(language="json", label="生成的 WaveDrom JSON 代码")
            
    gr.Markdown("### 🌊 实时渲染波形预览")
    output_html = gr.HTML(label="WaveDrom 渲染结果")
    
    submit_btn.click(
        fn=process_image, 
        inputs=input_image, 
        outputs=[output_code, output_html]
    )
    
    gr.Markdown("""
    ---
    **Note:** 首次运行可能需要一点时间加载模型。如果下方网页端渲染空白，您可以直接复制生成的 JSON 代码前往 [WaveDrom 官方在线编辑器](https://wavedrom.com/editor.html) 查看渲染效果。
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
