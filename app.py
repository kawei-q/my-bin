import os
import torch
import gradio as gr
from transformers import pipeline
from diffusers import StableDiffusionPipeline

# 如果需要使用 Hugging Face 访问令牌，取消下面两行的注释并设置环境变量 HUGGINGFACE_TOKEN
# from huggingface_hub import login
# login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Step 1: Prompt-to-Prompt 模块，使用 Flan - T5 生成结构化提示词
llm = pipeline(
    task="text2text-generation",
    model="google/flan-t5-large",
    device=0 if torch.cuda.is_available() else -1
)

# Step 2: 加载 Stable Diffusion 模型
# 移除无效的 revision 参数，仅使用 torch_dtype 加速加载
sd_v15 = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
sd_v15 = sd_v15.to("cuda" if torch.cuda.is_available() else "cpu")

sd_xl = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0"
)
sd_xl = sd_xl.to("cuda" if torch.cuda.is_available() else "cpu")

# 可选：语音输入模块，使用 Whisper
asr = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-base",
    device=0 if torch.cuda.is_available() else -1
)


def transcribe(audio_path):
    """
    对音频文件进行转录
    :param audio_path: 音频文件路径
    :return: 转录后的文本
    """
    text = asr(audio_path)["text"]
    return text


def generate(description, model_choice, guidance_scale, negative_prompt, style):
    """
    根据输入生成图像
    :param description: 文本描述
    :param model_choice: 选择的模型
    :param guidance_scale: 引导比例
    :param negative_prompt: 反向提示词
    :param style: 选择的风格
    :return: 生成的提示词和图像
    """
    # 构造给 LLM 的指令
    instruction = (
        f"请将以下简短描述扩展为 Stable Diffusion 友好的提示词，包含细节和风格：\n"
        f"描述: '{description}'\n"
        f"风格: '{style}'"
    )
    result = llm(instruction, max_length=128)[0]["generated_text"].strip()
    prompt = result

    # 根据模型选择生成图像
    pipeline_model = sd_xl if model_choice == "SDXL" else sd_v15
    image = pipeline_model(
        prompt,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt
    ).images[0]
    return prompt, image


# Step 3: 构建 Gradio 界面
with gr.Blocks(title="Prompt-to-Image Generator") as demo:
    gr.Markdown("## 基于 LLM 的提示词生成与 Stable Diffusion 图像生成")

    with gr.Row():
        with gr.Column():
            desc_input = gr.Textbox(label="文本描述", placeholder="例如：空中的魔法树屋")
            style_dropdown = gr.Dropdown(
                choices=["幻想风格", "赛博朋克", "写实主义"],
                label="选择风格"
            )
            model_radio = gr.Radio(
                choices=["SD v1.5", "SDXL"],
                value="SD v1.5",
                label="选择模型"
            )
            guidance_slider = gr.Slider(
                minimum=0, maximum=20, step=0.5, value=7.5,
                label="Guidance Scale"
            )
            neg_text = gr.Textbox(
                label="反向提示词",
                placeholder="排除内容（如：低分辨率、水印）"
            )
            use_voice = gr.Checkbox(label="启用语音输入（加分项）")
            audio_input = gr.Audio(type="filepath", label="语音输入")
            generate_btn = gr.Button("生成图像")

        with gr.Column():
            prompt_output = gr.Textbox(label="生成的提示词")
            image_output = gr.Image(label="生成的图像")

    # 绑定语音转文字（仅当启用时）
    def conditional_transcribe(audio_path, use_voice_flag):
        return transcribe(audio_path) if use_voice_flag else None

    audio_input.change(
        fn=conditional_transcribe,
        inputs=[audio_input, use_voice],
        outputs=desc_input
    )

    # 点击按钮生成提示词并绘图
    generate_btn.click(
        fn=generate,
        inputs=[desc_input, model_radio, guidance_slider, neg_text, style_dropdown],
        outputs=[prompt_output, image_output]
    )

# Step 4: 启动应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    
    