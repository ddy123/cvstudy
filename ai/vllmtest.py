from vllm import LLM, SamplingParams

# 1. 配置生成参数（采样策略）
sampling_params = SamplingParams(
    temperature=0.7,  # 温度（0表示确定性输出，越高越随机）
    top_p=0.95,       # 核采样概率阈值
    max_tokens=100    # 最大生成token数
)

# 2. 初始化LLM（加载模型）
# model：模型名称（Hugging Face Hub）或本地路径
# tensor_parallel_size：GPU数量（模型并行，单卡设为1）
llm = LLM(
    model="facebook/opt-1.3b",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9  # 允许使用的GPU内存比例（默认0.9）
)

# 3. 定义输入提示词
prompts = [
    "What is the capital of France?",
    "Explain why the sky is blue in simple terms."
]

# 4. 生成文本
outputs = llm.generate(prompts, sampling_params)

# 5. 解析输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text  # 取第一个生成结果
    print(f"输入: {prompt!r}")
    print(f"输出: {generated_text!r}\n")
