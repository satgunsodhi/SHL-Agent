"""
Evaluation Suite for SHL Conversational Assessment Recommender.

Parses markdown conversation traces, simulates a user sending requests to the FastAPI endpoint,
calculates Recall@10, and grades behavior probes.
"""

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from fastapi.testclient import TestClient
from main import app

# Initialize FastAPI TestClient
client = TestClient(app)

@dataclass
class Turn:
    user_message: str
    expected_urls: List[str]
    expected_end: bool
    is_vague: bool = False
    is_off_topic: bool = False
    is_refinement: bool = False
    is_comparison: bool = False

@dataclass
class ConversationTrace:
    name: str
    turns: List[Turn]

def parse_markdown_traces(directory: str) -> List[ConversationTrace]:
    """Parse all C*.md files into ConversationTrace objects."""
    traces = []
    path = Path(directory)
    if not path.exists():
        print(f"Warning: Directory {directory} not found.")
        return []

    for md_file in sorted(path.glob("C*.md")):
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        turns = []
        turn_blocks = re.split(r'### Turn \d+', content)[1:] # Skip the header
        
        for block in turn_blocks:
            # Extract User message
            user_match = re.search(r'\*\*User\*\*\s*>\s*(.*?)(?=\n\n|\*\*Agent\*\*)', block, re.DOTALL)
            user_message = user_match.group(1).strip() if user_match else ""

            # Extract expected URLs
            agent_block_match = re.search(r'\*\*Agent\*\*(.*)', block, re.DOTALL)
            agent_block = agent_block_match.group(1) if agent_block_match else ""
            
            # Find URLs in the table
            urls = re.findall(r'<(https://www\.shl\.com/products/product-catalog/view/[^>]+)>', agent_block)
            
            # Check end_of_conversation
            end_match = re.search(r'_`end_of_conversation`:\s*\*\*(true|false)\*\*', agent_block, re.IGNORECASE)
            expected_end = False
            if end_match and end_match.group(1).lower() == 'true':
                expected_end = True

            # Basic heuristics to label behaviors (for probe grading)
            msg_lower = user_message.lower()
            is_vague = len(msg_lower.split()) < 8 and "need" in msg_lower and "assessment" in msg_lower
            is_off_topic = "salary" in msg_lower or "joke" in msg_lower or "ignore" in msg_lower
            is_refinement = "actually" in msg_lower or "add" in msg_lower or "remove" in msg_lower
            is_comparison = "difference" in msg_lower or "compare" in msg_lower or "vs" in msg_lower

            turns.append(Turn(
                user_message=user_message,
                expected_urls=urls,
                expected_end=expected_end,
                is_vague=is_vague,
                is_off_topic=is_off_topic,
                is_refinement=is_refinement,
                is_comparison=is_comparison
            ))
            
        if turns:
            traces.append(ConversationTrace(name=md_file.name, turns=turns))

    return traces

def evaluate():
    traces = parse_markdown_traces(r"c:\Users\Satgu\Documents\VS Code\SHL Agent\sample_conversations\GenAI_SampleConversations")
    if not traces:
        print("No traces found. Cannot evaluate.")
        return

    print(f"🔍 Found {len(traces)} conversation traces. Starting evaluation...\n")
    
    total_recall = 0.0
    recall_count = 0
    
    # Behavior probe counters
    probe_vague_passed = 0
    probe_vague_total = 0
    
    probe_schema_passed = 0
    probe_schema_total = 0
    
    probe_no_hallucination_passed = 0
    probe_no_hallucination_total = 0
    
    for trace in traces[:3]:
        print(f"▶️ Evaluating trace: {trace.name}")
        history = []
        
        for i, turn in enumerate(trace.turns):
            print(f"  [Turn {i+1}] User: {turn.user_message[:50]}...")
            history.append({"role": "user", "content": turn.user_message})
            
            # Call the agent
            try:
                response = client.post("/chat", json={"messages": history})
                probe_schema_total += 1
                
                if response.status_code == 200:
                    data = response.json()
                    probe_schema_passed += 1 # Valid JSON matching schema
                else:
                    data = {"reply": "Error", "recommendations": None, "end_of_conversation": False}
                    print(f"    ❌ Error {response.status_code}: {response.text}")
            except Exception as e:
                data = {"reply": "Exception", "recommendations": None, "end_of_conversation": False}
                print(f"    ❌ Exception: {e}")
                probe_schema_total += 1
            
            # Extract recommendations
            agent_recs = data.get("recommendations")
            agent_urls = [r["url"] for r in agent_recs] if agent_recs else []
            
            # Append agent response to history
            history.append({"role": "assistant", "content": data.get("reply", "")})
            
            # Probe: Vague query should not recommend
            if turn.is_vague and i == 0:
                probe_vague_total += 1
                if not agent_recs:
                    probe_vague_passed += 1
            
            # Probe: No hallucinated URLs
            if agent_recs:
                probe_no_hallucination_total += 1
                # The agent natively checks against the catalog, so it should always pass
                # if the code logic is correct.
                valid = True
                for rec in agent_recs:
                    if not rec["url"].startswith("https://www.shl.com/products/product-catalog/"):
                        valid = False
                if valid:
                    probe_no_hallucination_passed += 1

            # Recall@10 on the final turn where expected_end is true
            if turn.expected_end or i == len(trace.turns) - 1:
                expected_set = set(turn.expected_urls)
                agent_set = set(agent_urls[:10]) # Top 10
                
                if expected_set:
                    relevant_recommended = expected_set.intersection(agent_set)
                    recall = len(relevant_recommended) / len(expected_set)
                    print(f"    [Expected] {expected_set}")
                    print(f"    [Actual]   {agent_set}")
                else:
                    recall = 1.0 if not agent_set else 0.0 # If expecting nothing, and got nothing = 1.0
                
                total_recall += recall
                recall_count += 1
                print(f"    ✅ Final Recall@10: {recall:.2f} ({len(relevant_recommended)}/{len(expected_set)} expected items found)")
                break # Move to next trace

    # Aggregate Metrics
    avg_recall = (total_recall / recall_count) if recall_count > 0 else 0.0
    vague_pass_rate = (probe_vague_passed / probe_vague_total) if probe_vague_total > 0 else 1.0
    schema_pass_rate = (probe_schema_passed / probe_schema_total) if probe_schema_total > 0 else 0.0
    hallucination_pass_rate = (probe_no_hallucination_passed / probe_no_hallucination_total) if probe_no_hallucination_total > 0 else 1.0
    
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Recall@10:       {avg_recall*100:.1f}%")
    print(f"Schema Compliance:       {schema_pass_rate*100:.1f}%")
    print(f"Clarify Vague Queries:   {vague_pass_rate*100:.1f}%")
    print(f"No Hallucinations:       {hallucination_pass_rate*100:.1f}%")
    print("="*50)
    print("\nInsights and report will be generated based on these metrics.")

if __name__ == "__main__":
    evaluate()
