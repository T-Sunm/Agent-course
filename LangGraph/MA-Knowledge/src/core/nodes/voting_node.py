from typing import Dict, Any, List, Tuple
from collections import Counter
import re
# from src.core.state import ViReAgentState

def extract_answer_from_result(result: str) -> str:
    """
    Extract the actual answer from agent result text.
    Assumes the answer is the last word/phrase after parsing.
    """
    if not result:
        return ""
    
    # Try to extract answer after "Answer:" pattern
    if "Answer:" in result:
        answer_part = result.split("Answer:")[-1].strip()
        # Get the first word/phrase as the answer
        answer = answer_part.split()[0] if answer_part.split() else ""
        return answer.lower().strip('.,!?;:"')
    
    # If no "Answer:" pattern, use the last sentence as answer
    sentences = result.strip().split('.')
    if sentences:
        answer = sentences[-1].strip()
        # Get the first word as answer if it's a short phrase
        if len(answer.split()) <= 3:
            return answer.lower().strip('.,!?;:"')
    
    # Fallback: use the whole result if it's short enough
    if len(result.split()) <= 3:
        return result.lower().strip('.,!?;:"')
    
    return ""

def voting_function(junior_answer: str, senior_answer: str, manager_answer: str) -> Tuple[str, Dict[str, int]]:
    """
    Weighted voting function implementing AF = Voting(AJ[w1], AS[w2], AM[w3])
    
    Args:
        junior_answer: Answer from Junior agent (weight = 2)
        senior_answer: Answer from Senior agent (weight = 3)  
        manager_answer: Answer from Manager agent (weight = 4)
        
    Returns:
        Tuple of (final_answer, vote_breakdown)
    """
    # Define weights according to paper
    weights = {
        'junior': 2,
        'senior': 3,
        'manager': 4
    }
    
    # Count votes for each unique answer
    vote_counts = Counter()
    
    # Add weighted votes
    if junior_answer:
        vote_counts[junior_answer] += weights['junior']
        
    if senior_answer:
        vote_counts[senior_answer] += weights['senior']
        
    if manager_answer:
        vote_counts[manager_answer] += weights['manager']
    
    if vote_counts:
        final_answer = vote_counts.most_common(1)[0][0]
        return final_answer, dict(vote_counts)
    
    # Fallback if no valid answers
    return "", {}

def voting_node(state) -> Dict[str, Any]:
    """
    Voting node that implements weighted voting mechanism from paper.
    
    Process:
    1. Extract answers from each agent's results
    2. Apply weighted voting: Junior(2), Senior(3), Manager(4)
    3. Select answer with highest vote count
    4. Return final answer and voting details
    """
    
    print("=" * 50)
    print("VOTING PROCESS STARTED")
    print("=" * 50)
    
    # Extract results from agents
    # Note: results are accumulated in order [junior, senior, manager]
    results = state.get("results", [])
    
    if len(results) < 3:
        print(f"Warning: Expected 3 agent results, got {len(results)}")
        return {
            "final_answer": "",
            "voting_details": {
                "error": f"Insufficient results: expected 3, got {len(results)}"
            }
        }
    
    # convert list of dict to dict
    agent_results = {k: v for d in results for k, v in d.items()}
        
    # Extract answers from each agent by name
    junior_result = agent_results.get("Junior", "")
    senior_result = agent_results.get("Senior", "")
    manager_result = agent_results.get("Manager", "")
    
    junior_answer = extract_answer_from_result(junior_result)
    senior_answer = extract_answer_from_result(senior_result)
    manager_answer = extract_answer_from_result(manager_result)
    
    print(f"Junior Agent Answer: '{junior_answer}' (Weight: 2)")
    print(f"Senior Agent Answer: '{senior_answer}' (Weight: 3)")
    print(f"Manager Agent Answer: '{manager_answer}' (Weight: 4)")
    
    # Apply weighted voting
    final_answer, vote_breakdown = voting_function(
        junior_answer, senior_answer, manager_answer
    )
    
    # Create detailed voting information
    voting_details = {
        "agent_answers": {
            "junior": {"answer": junior_answer, "weight": 2},
            "senior": {"answer": senior_answer, "weight": 3},
            "manager": {"answer": manager_answer, "weight": 4}
        },
        "vote_breakdown": vote_breakdown,
        "final_answer": final_answer,
        "total_votes": sum(vote_breakdown.values()) if vote_breakdown else 0
    }
    
    print("\nVOTING BREAKDOWN:")
    for answer, votes in vote_breakdown.items():
        print(f"  Answer '{answer}': {votes} votes")
    
    print(f"\nFINAL SELECTED ANSWER: '{final_answer}'")
    print("=" * 50)
    
    return {
        "final_answer": final_answer,
        "voting_details": voting_details
    }

def weighted_voting_example():
    """
    Example demonstrating the voting mechanism with realistic data structure.
    
    Simulates the actual data structure after all agents complete their analysis.
    Example scenario:
    - Junior says "cat": 2 votes
    - Senior says "dog": 3 votes  
    - Manager says "dog": 4 votes
    → "dog" wins with 7 votes vs "cat" with 2 votes
    """
    
    print("REALISTIC VOTING EXAMPLE:")
    print("-" * 40)
    
    # Simulate realistic state after graph completion
    # This mimics what voting_node actually receives
    dummy_state = {
        "results": [
            {"Junior": "cat"},
            {"Senior": "dog"}, 
            {"Manager": "dog"}
        ]
    }
    
    print("Simulated results from agents:")
    for i, result_dict in enumerate(dummy_state["results"]):
        for agent_name, response in result_dict.items():
            print(f"  {agent_name}: '{response}'")
    
    # Process using actual voting_node logic
    print("\nProcessing through voting_node logic:")
    
    # Convert list of dict to dict (same as voting_node)
    agent_results = {k: v for d in dummy_state["results"] for k, v in d.items()}
    
    # Extract answers (same logic as voting_node)
    junior_result = agent_results.get("Junior", "")
    senior_result = agent_results.get("Senior", "")
    manager_result = agent_results.get("Manager", "")
    
    junior_answer = extract_answer_from_result(junior_result)
    senior_answer = extract_answer_from_result(senior_result)
    manager_answer = extract_answer_from_result(manager_result)
    
    print(f"  Extracted Junior answer: '{junior_answer}' (Weight: 2)")
    print(f"  Extracted Senior answer: '{senior_answer}' (Weight: 3)")
    print(f"  Extracted Manager answer: '{manager_answer}' (Weight: 4)")
    
    # Apply weighted voting
    final_answer, vote_breakdown = voting_function(
        junior_answer, senior_answer, manager_answer
    )
    
    print(f"\nVote breakdown: {vote_breakdown}")
    print(f"Winner: '{final_answer}' with {vote_breakdown.get(final_answer, 0)} total votes")
    print("-" * 40)
    
    return final_answer, vote_breakdown

# if __name__ == "__main__":
#     # Run example
#     weighted_voting_example()
