"""
Test script for LatentMAS API

Usage:
    1. Start the server: python server.py --model_name Qwen/Qwen2.5-7B-Instruct
    2. Run this test: python test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"

question = "A train travels from city A to city B at 60 km/h. The return trip is made at 40 km/h. What is the average speed for the entire journey?"
system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

debug_max_tokens = 1024  # Number of tokens to generate for debug preview in latent mode
debug_continuation_prompt = None

def test_health():
    """Test health endpoint."""
    resp = requests.get(f"{BASE_URL}/health")
    print("=== Health Check ===")
    print(json.dumps(resp.json(), indent=2))
    return resp.status_code == 200


def test_normal_mode():
    """Test normal generation mode."""
    print("\n=== Normal Mode ===")
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "mode": "normal",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ],
            "max_tokens": 1024,
        }
    )
    result = resp.json()
    print(f"Response: {result['choices'][0]['message']['content']}")
    print(f"Usage: {result['usage']}")
    return resp.status_code == 200


def test_latent_sequential():
    """Test latent mode with sequential agent calls."""
    print("\n=== Latent Sequential Mode (Fixed Steps) ===")
        
    # Step 1: Planner agent
    print("\n[Step 1] Planner Agent - Latent (5 steps)")
    planner_prompt = f"""You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

Question: {question}

Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.
Now output your plan to solve the question below:"""
    
    resp1 = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "mode": "latent",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": planner_prompt}
            ],
            "latent_steps": 5,
            "debug_max_tokens": debug_max_tokens,
            "debug_continuation_prompt": debug_continuation_prompt,
        }
    )
    result1 = resp1.json()
    session_id = result1["session_id"]
    print(f"Session ID: {session_id}")
    print(f"Content: {result1['choices'][0]['message']['content']}")
    print(f"Usage: {result1['usage']}")
    
    # Step 2: Critic agent
    print("\n[Step 2] Critic Agent - Latent (5 steps)")
    critic_prompt = f"""Question: {question}

You are a Critic Agent to evaluate the correctness of the input plan for the given question and provide helpful feedback for improving the plan.
The plan information is provided in latent KV representation format. Review the plan and question and output:
(1) original plan contents
(2) constructive feedback on the original plan.

Format your response as follows:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:"""
    
    resp2 = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "mode": "latent",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": critic_prompt}
            ],
            "session_id": session_id,
            "latent_steps": 5,
            "debug_max_tokens": debug_max_tokens,
            "debug_continuation_prompt": debug_continuation_prompt,
        }
    )
    result2 = resp2.json()
    session_id = result2["session_id"]
    print(f"Session ID: {result2['session_id']}")
    print(f"Content: {result2['choices'][0]['message']['content']}")
    print(f"Usage: {result2['usage']}")
    
    # Step 3: Refiner agent
    print("\n[Step 3] Refiner Agent - Latent (5 steps)")
    refiner_prompt = f"""Question: {question}

You are a Refiner Agent to provide a refined step-by-step plan for solving the given question.
You are provided with:
(1) latent-format information: a previous plan with feedback
(2) text-format information: the input question you need to solve.

Based on the input, write a refined and improved plan to solve the question. Make sure your output plan is correct and concise.

Now, output your refined plan below:"""
    
    resp3 = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "mode": "latent",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": refiner_prompt}
            ],
            "session_id": session_id,
            "latent_steps": 5,
            "debug_max_tokens": debug_max_tokens,
            "debug_continuation_prompt": debug_continuation_prompt,
        }
    )
    result3 = resp3.json()
    session_id = result3["session_id"]
    print(f"Session ID: {result3['session_id']}")
    print(f"Message: {result3['choices'][0]['message']['content']}")
    
    # Step 4: Judger agent - text generation
    print("\n[Step 4] Judger Agent - Text Generation")
    judger_prompt = f"""Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.

Format your response as follows:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Copy the provided Critic Agent's feedback here]
Refined Plan: [Copy the provided Refiner Agent's refined plan here]
Answer: \\boxed{{YOUR_FINAL_ANSWER}}

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""
    
    resp4 = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "mode": "text",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": judger_prompt}
            ],
            "session_id": session_id,
            "max_tokens": 1024,
        }
    )
    result4 = resp4.json()
    print(f"Session ID: {result4['session_id']}")
    print(f"Response:\n{result4['choices'][0]['message']['content']}")
    print(f"Usage: {result4['usage']}")
    
    return resp4.status_code == 200


def test_latent_dynamic():
    """Test latent mode with dynamic stopping (no latent_steps specified)."""
    print("\n=== Latent Dynamic Mode (Until EOS) ===")
        
    # Single latent call without specifying latent_steps
    print("\n[Dynamic Latent] No latent_steps specified - will generate until EOS")
    prompt = f"""You are a helpful math assistant. Solve this problem step by step.

Question: {question}

Solution:"""
    
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "mode": "latent",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            # latent_steps not specified - will use dynamic stopping
            "max_tokens": 100,  # This becomes max_latent_steps
            "debug_max_tokens": debug_max_tokens,
            "debug_continuation_prompt": debug_continuation_prompt,
        }
    )
    result = resp.json()
    session_id = result["session_id"]
    print(f"Session ID: {result['session_id']}")
    print(f"Content: {result['choices'][0]['message']['content']}")
    print(f"Usage: {result['usage']}")
    
    # Now use the session to generate final text
    print("\n[Text Generation] Using cached KV from dynamic latent")
    resp2 = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "mode": "text",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Now provide the final answer:"}
            ],
            "session_id": session_id,
            "max_tokens": 1024,
        }
    )
    result2 = resp2.json()
    print(f"Final answer: {result2['choices'][0]['message']['content']}")
    
    return resp.status_code == 200


def test_session_management():
    """Test session management endpoints."""
    print("\n=== Session Management ===")
    
    # Create a session
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "mode": "latent",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Test message"}
            ],
            "latent_steps": 3,
        }
    )
    session_id = resp.json()["session_id"]
    print(f"Created session: {session_id}")
    
    # Check session exists
    resp = requests.get(f"{BASE_URL}/v1/sessions/{session_id}")
    print(f"Session exists: {resp.json()}")
    
    # List sessions
    resp = requests.get(f"{BASE_URL}/v1/sessions")
    print(f"Active sessions: {resp.json()}")
    
    # Delete session
    resp = requests.delete(f"{BASE_URL}/v1/sessions/{session_id}")
    print(f"Deleted session: {resp.json()}")
    
    # Verify deleted
    resp = requests.get(f"{BASE_URL}/v1/sessions/{session_id}")
    print(f"Session after delete: {resp.json()}")
    
    return True


def test_kv_injection_ablation():
    """Ablation test: verify KV cache injection by comparing outputs with/without cached KV."""
    print("\n=== KV Injection Ablation Test ===")
    
    # Step 1: Create latent context about a fictional character
    print("\n[Step 1] Creating latent context about a character...")
    context_prompt = """You are creating a character profile. Here is the information:

Name: Dr. Jay Voss
Occupation: Algorithm Engineer at Microsoft
Age: 42
Notable achievement: Developed a groundbreaking machine learning algorithm that significantly improved search engine efficiency.
Personality: Meticulous, introverted, loves classical music, and has a passion for solving complex problems.

Remember this character information for future reference."""
    
    resp1 = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "mode": "latent",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": context_prompt}
            ],
            "latent_steps": 5,
            "debug_max_tokens": debug_max_tokens,
        }
    )
    result1 = resp1.json()
    session_id = result1["session_id"]
    print(f"✓ Created session: {session_id}")
    print(f"  Latent steps: {result1['usage']['latent_steps']}")
    
    # Step 2: Test prompt that requires the cached context
    test_prompt = "List the Occupation, Age, Notable achievement and Personality of Dr. Jay Voss."
    
    # Step 2a: WITH cached KV (should know the answer)
    print("\n[Step 2a] Testing WITH cached KV (session_id provided)...")
    resp_with_kv = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "mode": "text",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": test_prompt}
            ],
            "session_id": session_id,
            "debug_max_tokens": debug_max_tokens,
            "debug_continuation_prompt": debug_continuation_prompt,
            "max_tokens": 150,
        }
    )
    result_with_kv = resp_with_kv.json()
    output_with_kv = result_with_kv['choices'][0]['message']['content']
    print(f"Response WITH KV:\n{output_with_kv}\n")
    
    # Step 2b: WITHOUT cached KV (should NOT know the answer)
    print("[Step 2b] Testing WITHOUT cached KV (no session_id)...")
    resp_without_kv = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "mode": "normal",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": test_prompt}
            ],
            "max_tokens": 150,
        }
    )
    result_without_kv = resp_without_kv.json()
    output_without_kv = result_without_kv['choices'][0]['message']['content']
    print(f"Response WITHOUT KV:\n{output_without_kv}\n")
    
    # Step 3: Analyze results
    print("=== ABLATION ANALYSIS ===")
    
    # Check if outputs are different
    outputs_differ = output_with_kv.strip() != output_without_kv.strip()
    print(f"✓ Outputs are different: {outputs_differ}")
    
    # Check if "with KV" response mentions key information
    has_voss = "voss" in output_with_kv.lower()
    has_algorithm_engineer = "algorithm engineer" in output_with_kv.lower()
    has_context_info = has_voss and has_algorithm_engineer
    print(f"✓ Response WITH KV contains context info: {has_context_info}")
    
    # Check if "without KV" response lacks the information
    lacks_specific_info = "voss" not in output_without_kv.lower() and "algorithm engineer" not in output_without_kv.lower()
    print(f"✓ Response WITHOUT KV lacks specific context: {lacks_specific_info}")
    
    # Overall verdict
    injection_working = outputs_differ and has_context_info and lacks_specific_info
    print(f"\n{'✓✓✓ KV INJECTION VERIFIED: Working correctly' if injection_working else '✗✗✗ KV INJECTION ISSUE: May not be working'}")
    
    if not injection_working:
        print("\nDEBUG INFO:")
        print(f"  Outputs differ: {outputs_differ}")
        print(f"  With KV has context: {has_context_info}")
        print(f"  Without KV lacks context: {lacks_specific_info}")
    
    # Cleanup
    requests.delete(f"{BASE_URL}/v1/sessions/{session_id}")
    
    return injection_working


if __name__ == "__main__":
    print("LatentMAS API Test\n")
    
    try:
        # Run tests
        test_health()
        test_normal_mode()
        test_latent_sequential()
        test_latent_dynamic()
        test_session_management()
        test_kv_injection_ablation()  # NEW: Verify KV injection is working
        
        print("\n=== All tests completed ===")
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server. Make sure the server is running:")
        print("  python server.py --model_name Qwen/Qwen2.5-7B-Instruct")
