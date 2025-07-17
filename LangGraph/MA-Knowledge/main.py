#!/usr/bin/env python3
"""
Main entry point for Visual Multi-Agent QA System
"""

from io import BytesIO
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from io import BytesIO
from src.core.graph_builder.main_graph import MainGraphBuilder
from src.tools.knowledge_tools import arxiv, wikipedia
from src.tools.vqa_tool import vqa_tool
from src.core.memory_manager import session_memory
from src.agents.strategies.junior_agent import create_junior_agent
from src.agents.strategies.senior_agent import create_senior_agent
from src.agents.strategies.manager_agent import create_manager_agent


def setup_tools_registry() -> Dict[str, Any]:
    """Setup tools registry"""
    return {
        "vqa_tool": vqa_tool,
        "arxiv": arxiv,
        "wikipedia": wikipedia, 
    }

def run_visual_qa(question: str, image_url: str, thread_id: str = "default") -> Dict[str, Any]:
    """
    Run visual question answering with multi-agent system
    
    Args:
        question: The question to ask about the image
        image_url: URL or path to the image
        thread_id: Thread ID for session management
        
    Returns:
        Dict containing the results from all analysts
    """
    
    # Setup
    tools_registry = setup_tools_registry()
    
    # Build graph
    builder = MainGraphBuilder(tools_registry, memory_enabled=True)
    graph = builder.create_main_workflow()
    
    # png_data = graph.get_graph().draw_mermaid_png()
    # img = Image.open(BytesIO(png_data))
    # plt.figure(figsize=(15, 10))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title('Workflow Graph Visualization', fontsize=16, fontweight='bold')
    # plt.tight_layout()
    # plt.show()
        
    # Prepare input
    initial_state = {
        "question": question,
        "image": image_url,
    }
    
    # Configure session
    thread_config = session_memory.create_thread_config(thread_id)
    
    print(f"Image: {image_url}")
    print(f"Question: {question}")
    print("-" * 50)
    
    final_state = graph.invoke(initial_state, config=thread_config)
    print("Analysis completed!")
    return final_state

def main():

    examples = [
        {
            "question": "What color is the dog's fur?",
            "image_url": "https://github.com/NVlabs/describe-anything/blob/main/images/1.jpg?raw=true"
        },
        {
            "question": "What breed might this dog be?",
            "image_url": "https://github.com/NVlabs/describe-anything/blob/main/images/1.jpg?raw=true"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i}")
        print(f"{'='*60}")
        
        result = run_visual_qa(
            question=example["question"],
            image_url=example["image_url"],
            thread_id=f"example_{i}"
        )
        
        print(f"\n📊 FINAL RESULTS:")
        for j, res in enumerate(result["results"], 1):
            print(f"{j}. {res}")

if __name__ == "__main__":
    main()