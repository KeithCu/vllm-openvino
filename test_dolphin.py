import time
from vllm import LLM, SamplingParams

# Start timing
start_time = time.time()

# Initialize LLM
llm = LLM(model="/home/keithcu/Desktop/Python/vllm-openvino/Dolphin3-ov", dtype="auto", max_model_len=8192)

# Generate
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
prompt = "Tell a funny story about Abbott and Costello buying a 3-d printer"
output = llm.generate(prompt, sampling_params)

# End timing
end_time = time.time()
total_time = end_time - start_time

# Calculate tokens per second (subtract 6 seconds for init time)
inference_time = max(0, total_time - 40.0)  # Ensure non-negative
generated_text = output[0].outputs[0].text
num_tokens = len(output[0].outputs[0].token_ids) if hasattr(output[0].outputs[0], 'token_ids') else len(output[0].outputs[0].text.split())
tokens_per_second = num_tokens / inference_time if inference_time > 0 else 0

# Print results
print("\n" + "=" * 60)
print("Generation Results")
print("=" * 60)
print(generated_text)
print("\n" + "=" * 60)
print("Performance Metrics")
print("=" * 60)
print(f"Total time: {total_time:.2f} seconds")
print(f"Inference time (total - 6s init): {inference_time:.2f} seconds")
print(f"Generated tokens: {num_tokens}")
print(f"Tokens per second: {tokens_per_second:.2f} tok/s")
print("=" * 60)
