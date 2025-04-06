#!/usr/bin/env python3
import asyncio
import nest_asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.ollama import Ollama

# Áp dụng nest_asyncio để hỗ trợ asyncio (nếu cần, ví dụ trong Jupyter)
nest_asyncio.apply()

# Định nghĩa một công cụ tính toán đơn giản
def multiply(a: float, b: float) -> float:
  """Useful for multiplying two numbers."""
  return a * b


# Khởi tạo Ollama LLM cho agent với model đã có trên Ollama server
llm_agent = Ollama(
    model="qwen2.5:0.5b",
    # Đảm bảo server Ollama đang chạy ở địa chỉ này
    base_url="http://127.0.0.1:12345",
    request_timeout=360.0
)

# Tạo workflow agent với công cụ calculator
agent = FunctionAgent(
    tools=[multiply],
    llm=llm_agent,
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)

async def main():
  # Chạy agent và in kết quả
  response = await agent.run("What is 1234 * 4567?")
  print("Agent Response:")
  print(response)

def run_agent():
  asyncio.run(main())


if __name__ == "__main__":
  run_agent()
