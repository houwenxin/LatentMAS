"""
Test script for LatentMAS API

Usage:
    1. Start the server: python server.py --model_name Qwen/Qwen2.5-7B-Instruct
    2. Run this test: python test_api.py
"""

import requests
import json

# BASE_URL = "http://localhost:8000"
BASE_URL = "http://10.224.120.13:8000"

question = "A train travels from city A to city B at 60 km/h. The return trip is made at 40 km/h. What is the average speed for the entire journey?"
system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

debug_max_tokens = None  # Number of tokens to generate for debug preview in latent mode
debug_continuation_prompt = None
latent_space_realign = False  # Whether to use latent space realignment in LatentMAS
latent_only = False
add_think_token = True

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
            "latent_space_realign": latent_space_realign,
            "debug_max_tokens": debug_max_tokens,
            "debug_continuation_prompt": debug_continuation_prompt,
            "latent_only": latent_only,
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
            "latent_space_realign": latent_space_realign,
            "debug_max_tokens": debug_max_tokens,
            "debug_continuation_prompt": debug_continuation_prompt,
            "latent_only": latent_only,
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
            "latent_space_realign": latent_space_realign,
            "debug_max_tokens": debug_max_tokens,
            "debug_continuation_prompt": debug_continuation_prompt,
            "latent_only": latent_only,
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
            "latent_space_realign": latent_space_realign,
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

Remember this character information in the first 20 output tokens for future reference."""
    
    resp1 = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "mode": "latent",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": context_prompt}
            ],
            "latent_steps": 20,
            "latent_space_realign": latent_space_realign,
            "debug_max_tokens": debug_max_tokens,
            "debug_continuation_prompt": debug_continuation_prompt,
            "latent_only": latent_only,
            "add_think_token": add_think_token,
        }
    )
    result1 = resp1.json()
    session_id = result1["session_id"]
    print(f"✓ Created session: {session_id}")
    print(f"  Latent steps: {result1['usage']['latent_steps']}")
    print(f"  Usage: {result1['usage']}")
    print(f"  Content: {result1['choices'][0]['message']['content']}")
    
    # Step 2: Test prompt that requires the cached context
    test_prompt = "Based on the input context, please list the Occupation, Age, Notable achievement and Personality of Dr. Jay Voss."
    
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
            "latent_space_realign": latent_space_realign,
            "debug_max_tokens": debug_max_tokens,
            "debug_continuation_prompt": debug_continuation_prompt,
            "max_tokens": 1024,
            "latent_only": latent_only,
            "add_think_token": add_think_token,
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
            "max_tokens": 1024,
            "add_think_token": add_think_token,
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
    lacks_specific_info = "algorithm engineer" not in output_without_kv.lower()
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


def test_complex_task():
    prompt = """<query>
    List all MacBook Air configurations with their prices from this page.
    </query>
    <current_url>
    https://www.apple.com/shop/buy-mac/macbook-air
    </current_url>

    <content_stats>
    Content processed: 1,324,938 HTML chars → 69,055 initial markdown → 68,055 filtered markdown (filtered 1,000 chars of noise)
    </content_stats>

    <webpage_content>
    ![](https://securemetrics.apple.com/b/ss/applestoreww/1/H.8--NS/0?pageName=No-Script:AOS%3A+home%2Fshop_mac%2Ffamily%2Fmacbook_air%2Fselect)
    * Apple
    *     * Store
        * ## Shop
        * Shop Gifts
        * Mac
        * iPad
        * iPhone
        * Apple Watch
        * Apple Vision Pro
        * AirPods
        * Accessories
    ## Quick Links
        * Find a Store
        * Order Status
        * Apple Trade In
        * Financing
        * Personal Setup
    ## Shop Special Stores
        * Certified Refurbished
        * Education
        * Business
        * Veterans and Military
        * Government
        * Mac
        * ## Explore Mac
        * Explore All Mac
        * MacBook Air
        * MacBook Pro
        * iMac
        * Mac mini
        * Mac Studio
        * Mac Pro
        * Displays
        * Compare Mac
        * Switch from PC to Mac
    ## Shop Mac
        * Shop Mac
        * Help Me Choose
        * Mac Accessories
        * Apple Trade In
        * Financing
    ## More from Mac
        * Mac Support
        * AppleCare
        * macOS Tahoe
        * Apple Intelligence
        * Apps by Apple
        * Better with iPhone
        * iCloud+
        * Mac for Business
        * Education
        * iPad
        * ## Explore iPad
        * Explore All iPad
        * iPad Pro
        * iPad Air
        * iPad
        * iPad mini
        * Apple Pencil
        * Keyboards
        * Compare iPad
    ## Shop iPad
        * Shop iPad
        * iPad Accessories
        * Apple Trade In
        * Financing
    ## More from iPad
        * iPad Support
        * AppleCare
        * iPadOS 26
        * Apple Intelligence
        * Apps by Apple
        * iCloud+
        * Education
        * iPhone
        * ## Explore iPhone
        * Explore All iPhone
        * iPhone 17 Pro
        * iPhone Air
        * iPhone 17
        * iPhone 16
        * iPhone 16e
        * Compare iPhone
        * Switch from Android
    ## Shop iPhone
        * Shop iPhone
        * iPhone Accessories
        * Apple Trade In
        * Carrier Deals at Apple
        * Financing
    ## More from iPhone
        * iPhone Support
        * AppleCare
        * iOS 26
        * Apple Intelligence
        * Apps by Apple
        * iPhone Privacy
        * Better with Mac
        * iCloud+
        * Wallet, Pay, Card
        * Siri
        * Watch
        * ## Explore Watch
        * Explore All Apple Watch
        * Apple Watch Series 11
        * Apple Watch SE 3
        * Apple Watch Ultra 3
        * Apple Watch Nike
        * Apple Watch Hermès
        * Compare Watch
        * Why Apple Watch
    ## Shop Watch
        * Shop Apple Watch
        * Apple Watch Bands
        * Apple Watch Accessories
        * Apple Trade In
        * Financing
    ## More from Watch
        * Apple Watch Support
        * AppleCare
        * watchOS 26
        * Apple Watch For Your Kids
        * Apps by Apple
        * Apple Fitness+
        * Vision
        * ## Explore Vision
        * Explore Apple Vision Pro
        * Tech Specs
    ## Shop Vision
        * Shop Apple Vision Pro
        * Apple Vision Pro Accessories
        * Book a Demo
        * Financing
    ## More from Vision
        * Apple Vision Pro Support
        * AppleCare
        * visionOS 26
        * AirPods
        * ## Explore AirPods
        * Explore All AirPods
        * AirPods 4
        * AirPods Pro 3
        * AirPods Max
        * Compare AirPods
    ## Shop AirPods
        * Shop AirPods
        * AirPods Accessories
    ## More from AirPods
        * AirPods Support
        * AppleCare
        * Hearing Health
        * Apple Music
        * Apple Fitness+
        * TV & Home
        * ## Explore TV & Home
        * Explore TV & Home
        * Apple TV 4K
        * HomePod
        * HomePod mini
    ## Shop TV & Home
        * Shop Apple TV 4K
        * Shop HomePod
        * Shop HomePod mini
        * Shop Siri Remote
        * TV & Home Accessories
    ## More from TV & Home
        * Apple TV Support
        * HomePod Support
        * AppleCare for Apple TV
        * AppleCare for HomePod
        * Apple TV app
        * Apple TV
        * Home app
        * Apple Music
        * Siri
        * AirPlay
        * Entertainment
        * ## Explore Entertainment
        * Explore Entertainment
        * Apple One
        * Apple TV
        * Apple Music
        * Apple Arcade
        * Apple Fitness+
        * Apple News+
        * Apple Podcasts
        * Apple Books
        * App Store
    ## Support
        * Apple TV Support
        * Apple Music Support
        * Accessories
        * ## Shop Accessories
        * Shop All Accessories
        * Mac
        * iPad
        * iPhone
        * Apple Watch
        * Apple Vision Pro
        * AirPods
        * TV & Home
    ## Explore Accessories
        * Made by Apple
        * Beats
        * AirTag
        * Assistive Technologies
        * Support
        * ## Explore Support
        * iPhone
        * Mac
        * iPad
        * Watch
        * Apple Vision Pro
        * AirPods
        * Music
        * TV
        * Explore Support
    ## Get Help
        * Community
        * Check Coverage
        * Genius Bar
        * Repair
    ## Helpful Topics
        * Get AppleCare
        * Apple Account and Password
        * Billing & Subscriptions
        * Accessibility
    * ## Quick Links
        * Shop Gifts
        * Find a Store
        * Apple Gift Card
        * Apple Vision Pro
        * Apple Trade In
    * 0+
    MacBook Air 
    * Overview 
    * macOS 
    * Compare 
    Pay for your new Mac over 12 months at 0% APR with Apple Card.Footnote◊ Just choose Apple Card Monthly Installments when you check out at Apple. Learn more
    # Choose your new MacBook Air.
    Have questions about buying a Mac?  
    Chat with a Mac Specialist(Opens in a new window)
    Select a size:
    13-inchFrom $999
    15-inchFrom $1199
    ![13-inch MacBook Air, top open, Liquid Retina display, rounded corners, raised feet, Sky Blue color](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-skyblue-select-202503?wid=904&hei=840&fmt=jpeg&qlt=90&.v=M2RyY09CWXlTQUp1KzEveHR6VXNxcTQ1bzN1SitYTU83Mm9wbk1xa1lWNC9UNzNvY2N5NXJTTDQ2YkVYYmVXakJkRlpCNVhYU3AwTldRQldlSnpRa0lIV0Fmdk9rUlVsZ3hnNXZ3K3lEVlk)Select a finish M4 chip with 10-core CPU 8-core GPU Processor 256GB Storage 16GB memorySky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-skyblue-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=M2RyY09CWXlTQUp1KzEveHR6VXNxckZUK08zZUpDWjVkdThxRjFOeXpiZ1loZk12ZmYvUVdUWXN6enVNb1ZMV1lJb0k2dU9zNW9xMXJqZDBvM2Qyb0xTNE1pQi9QM1JHSG9jeVpzRWk0Y3BTeVdNUUpkdUNVQmFQWUtkeTlaakc)Sky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-silver-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=cCtGdk55eHp5QXJpcHVMMTl2dWZRVTd6N3NGS052Vk8vV1djRWEyczF6enBnUExpanZJKzYrcTVScDVOQkk1S0NBd3lOUFpnTTVCeDVDYzlNNEhMcGVQdVlGd3VrOGlVYkpnMGk1RGpNTVg0TjVYSTlBSTJEWFRmd2Q1ejdzcUw)Silver
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-starlight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=U3RnbTBEaGs0WGRSa05FY2NWZjZNYnRZWE8yTitQUnBsaDIyYUxJdEFnMjRtQkh1eFY2UEs1YzJiZnNMZGs4LzBnYTc5dzVwRHNGRnI0TUhlVTZ5M01oWmc2WllwMXFwcWRvdkhKZ2lpU2F1dFJrcTkxU3lYM0RQR3RSN3V2bGs)Starlight
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-midnight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=MWFRUnJZcWcrd0xyc0tIOUdIbnJlK0duSURuNUhkK2pOZnE1VnZHT1ZSWHZLN2M3ZGNkS1NNWStVS2JNbmRmN2RobUVKV0dYZHJrUWxHbk5lRDZzaFVtVDkrYjh1cXBHVkh2dmRxL2dHRkZFYVdKMW9LeXZTaXRrdkpJWktJcng)Midnight
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    8-Core GPU  
    16GB Unified Memory  
    256GB SSD Storage  footnote  ¹
    * 16-core Neural Engine
    * 13.6-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 30W USB-C Power Adapter
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/store-feature-mac-btr-appleintelligence?wid=40&hei=40&fmt=png-alpha&.v=L2k0UWhTOS9yRDNpVUx5cHpsQmZ2QVFRNnlzRVVmQlhLLy9yOFBWUWhWUjRCdVRQdmxzSUpzNHlRVmliMmtQYlEwTnpsQWxmZlludFcyRm4wOWcxaTFtcXBPRUNsRkE2bCtPZmcxeHdTcnpHbDRFMFhQRUtYQ25zNWRCb3liYzQ)Apple Intelligence Footnote ∆
    $999.00
    $83.25/mo.per month for 12 mo.monthsFootnote *
    Get 3% Daily Cash with Apple Card
    ### Apple Trade In
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/tradein-icon-currency?wid=50&hei=50&fmt=png-alpha&.v=SHZ4TEtxLzFWZTROQ21qbUU5QnN6WFpLVyt6cGFLME5kV2hjK3Q2TmdSdXpCNjVQRjg4dWNzbjlzU2FEVW1Qb25hT3pZVkRHQkc0Sk1OWE5mMzlJR3NzS0hkb3pGTXJoWThTNkIrNkxEaUU)
    ### Apple Trade In
    Get credit toward a new Mac when you trade in your eligible computer. Or recycle it for free.Footnote◊◊
    How does trade-in work?
    Get started with trade inM4 chip with 10-core CPU 8-core GPU Processor 256GB Storage 16GB memory
    SelectM4 chip with 10-core CPU 8-core GPU Processor 256GB Storage 16GB memory 
    ### Need a moment?
    Keep all your selections by saving this device to Your Saves, then come back anytime and pick up right where you left off.
    Save for later
    ![13-inch MacBook Air, top open, Liquid Retina display, rounded corners, raised feet, Sky Blue color](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-skyblue-select-202503?wid=904&hei=840&fmt=jpeg&qlt=90&.v=M2RyY09CWXlTQUp1KzEveHR6VXNxcTQ1bzN1SitYTU83Mm9wbk1xa1lWNC9UNzNvY2N5NXJTTDQ2YkVYYmVXakJkRlpCNVhYU3AwTldRQldlSnpRa0lIV0Fmdk9rUlVsZ3hnNXZ3K3lEVlk)Select a finish M4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 16GB memorySky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-skyblue-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=M2RyY09CWXlTQUp1KzEveHR6VXNxckZUK08zZUpDWjVkdThxRjFOeXpiZ1loZk12ZmYvUVdUWXN6enVNb1ZMV1lJb0k2dU9zNW9xMXJqZDBvM2Qyb0xTNE1pQi9QM1JHSG9jeVpzRWk0Y3BTeVdNUUpkdUNVQmFQWUtkeTlaakc)Sky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-silver-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=cCtGdk55eHp5QXJpcHVMMTl2dWZRVTd6N3NGS052Vk8vV1djRWEyczF6enBnUExpanZJKzYrcTVScDVOQkk1S0NBd3lOUFpnTTVCeDVDYzlNNEhMcGVQdVlGd3VrOGlVYkpnMGk1RGpNTVg0TjVYSTlBSTJEWFRmd2Q1ejdzcUw)Silver
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-starlight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=U3RnbTBEaGs0WGRSa05FY2NWZjZNYnRZWE8yTitQUnBsaDIyYUxJdEFnMjRtQkh1eFY2UEs1YzJiZnNMZGs4LzBnYTc5dzVwRHNGRnI0TUhlVTZ5M01oWmc2WllwMXFwcWRvdkhKZ2lpU2F1dFJrcTkxU3lYM0RQR3RSN3V2bGs)Starlight
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-midnight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=MWFRUnJZcWcrd0xyc0tIOUdIbnJlK0duSURuNUhkK2pOZnE1VnZHT1ZSWHZLN2M3ZGNkS1NNWStVS2JNbmRmN2RobUVKV0dYZHJrUWxHbk5lRDZzaFVtVDkrYjh1cXBHVkh2dmRxL2dHRkZFYVdKMW9LeXZTaXRrdkpJWktJcng)Midnight
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    16GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    * 16-core Neural Engine
    * 13.6-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/store-feature-mac-btr-appleintelligence?wid=40&hei=40&fmt=png-alpha&.v=L2k0UWhTOS9yRDNpVUx5cHpsQmZ2QVFRNnlzRVVmQlhLLy9yOFBWUWhWUjRCdVRQdmxzSUpzNHlRVmliMmtQYlEwTnpsQWxmZlludFcyRm4wOWcxaTFtcXBPRUNsRkE2bCtPZmcxeHdTcnpHbDRFMFhQRUtYQ25zNWRCb3liYzQ)Apple Intelligence Footnote ∆
    $1,199.00
    $99.91/mo.per month for 12 mo.monthsFootnote *
    Get 3% Daily Cash with Apple Card
    ### Apple Trade In
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/tradein-icon-currency?wid=50&hei=50&fmt=png-alpha&.v=SHZ4TEtxLzFWZTROQ21qbUU5QnN6WFpLVyt6cGFLME5kV2hjK3Q2TmdSdXpCNjVQRjg4dWNzbjlzU2FEVW1Qb25hT3pZVkRHQkc0Sk1OWE5mMzlJR3NzS0hkb3pGTXJoWThTNkIrNkxEaUU)
    ### Apple Trade In
    Get credit toward a new Mac when you trade in your eligible computer. Or recycle it for free.Footnote◊◊
    How does trade-in work?
    Get started with trade inM4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 16GB memory
    SelectM4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 16GB memory 
    ### Need a moment?
    Keep all your selections by saving this device to Your Saves, then come back anytime and pick up right where you left off.
    Save for later
    ![13-inch MacBook Air, top open, Liquid Retina display, rounded corners, raised feet, Sky Blue color](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-skyblue-select-202503?wid=904&hei=840&fmt=jpeg&qlt=90&.v=M2RyY09CWXlTQUp1KzEveHR6VXNxcTQ1bzN1SitYTU83Mm9wbk1xa1lWNC9UNzNvY2N5NXJTTDQ2YkVYYmVXakJkRlpCNVhYU3AwTldRQldlSnpRa0lIV0Fmdk9rUlVsZ3hnNXZ3K3lEVlk)Select a finish M4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 24gb memorySky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-skyblue-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=M2RyY09CWXlTQUp1KzEveHR6VXNxckZUK08zZUpDWjVkdThxRjFOeXpiZ1loZk12ZmYvUVdUWXN6enVNb1ZMV1lJb0k2dU9zNW9xMXJqZDBvM2Qyb0xTNE1pQi9QM1JHSG9jeVpzRWk0Y3BTeVdNUUpkdUNVQmFQWUtkeTlaakc)Sky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-silver-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=cCtGdk55eHp5QXJpcHVMMTl2dWZRVTd6N3NGS052Vk8vV1djRWEyczF6enBnUExpanZJKzYrcTVScDVOQkk1S0NBd3lOUFpnTTVCeDVDYzlNNEhMcGVQdVlGd3VrOGlVYkpnMGk1RGpNTVg0TjVYSTlBSTJEWFRmd2Q1ejdzcUw)Silver
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-starlight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=U3RnbTBEaGs0WGRSa05FY2NWZjZNYnRZWE8yTitQUnBsaDIyYUxJdEFnMjRtQkh1eFY2UEs1YzJiZnNMZGs4LzBnYTc5dzVwRHNGRnI0TUhlVTZ5M01oWmc2WllwMXFwcWRvdkhKZ2lpU2F1dFJrcTkxU3lYM0RQR3RSN3V2bGs)Starlight
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-midnight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=MWFRUnJZcWcrd0xyc0tIOUdIbnJlK0duSURuNUhkK2pOZnE1VnZHT1ZSWHZLN2M3ZGNkS1NNWStVS2JNbmRmN2RobUVKV0dYZHJrUWxHbk5lRDZzaFVtVDkrYjh1cXBHVkh2dmRxL2dHRkZFYVdKMW9LeXZTaXRrdkpJWktJcng)Midnight
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    24GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    * 16-core Neural Engine
    * 13.6-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/store-feature-mac-btr-appleintelligence?wid=40&hei=40&fmt=png-alpha&.v=L2k0UWhTOS9yRDNpVUx5cHpsQmZ2QVFRNnlzRVVmQlhLLy9yOFBWUWhWUjRCdVRQdmxzSUpzNHlRVmliMmtQYlEwTnpsQWxmZlludFcyRm4wOWcxaTFtcXBPRUNsRkE2bCtPZmcxeHdTcnpHbDRFMFhQRUtYQ25zNWRCb3liYzQ)Apple Intelligence Footnote ∆
    $1,399.00
    $116.58/mo.per month for 12 mo.monthsFootnote *
    Get 3% Daily Cash with Apple Card
    ### Apple Trade In
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/tradein-icon-currency?wid=50&hei=50&fmt=png-alpha&.v=SHZ4TEtxLzFWZTROQ21qbUU5QnN6WFpLVyt6cGFLME5kV2hjK3Q2TmdSdXpCNjVQRjg4dWNzbjlzU2FEVW1Qb25hT3pZVkRHQkc0Sk1OWE5mMzlJR3NzS0hkb3pGTXJoWThTNkIrNkxEaUU)
    ### Apple Trade In
    Get credit toward a new Mac when you trade in your eligible computer. Or recycle it for free.Footnote◊◊
    How does trade-in work?
    Get started with trade inM4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 24gb memory
    SelectM4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 24gb memory 
    ### Need a moment?
    Keep all your selections by saving this device to Your Saves, then come back anytime and pick up right where you left off.
    Save for later
    ![15-inch MacBook Air, top open, Liquid Retina display, rounded corners, raised feet, Sky Blue color](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-skyblue-select-202503?wid=904&hei=840&fmt=jpeg&qlt=90&.v=REV4NmZ6SUhUbzJzVXZrcXZ3UGg2NjQ1bzN1SitYTU83Mm9wbk1xa1lWNC9UNzNvY2N5NXJTTDQ2YkVYYmVXakJkRlpCNVhYU3AwTldRQldlSnpRa0JGbFFCaXFWTk5QRkxaWFZ6TExmVXM)Select a finish M4 chip with 10-core CPU 10-core GPU Processor 256GB Storage 16GB memorySky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-skyblue-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=REV4NmZ6SUhUbzJzVXZrcXZ3UGg2N0ZUK08zZUpDWjVkdThxRjFOeXpiZ1loZk12ZmYvUVdUWXN6enVNb1ZMV1lJb0k2dU9zNW9xMXJqZDBvM2Qyb0xTNE1pQi9QM1JHSG9jeVpzRWk0Y3FOZkl3UzBsTFI3bGFSRmthOHVTSW4)Sky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-silver-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=YmVzOXFnNklRUk9jbTVLMFNybEcyRTd6N3NGS052Vk8vV1djRWEyczF6enBnUExpanZJKzYrcTVScDVOQkk1S0NBd3lOUFpnTTVCeDVDYzlNNEhMcFFPS3NYaFB3eWhNUXRVbVdTUWppalVKbndhbjJyQUQ2Q0NQdmRNNWZJWis)Silver
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-starlight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=TkljV1hUT21PWENUbGQ5c24rTjNVYnRZWE8yTitQUnBsaDIyYUxJdEFnMjRtQkh1eFY2UEs1YzJiZnNMZGs4LzBnYTc5dzVwRHNGRnI0TUhlVTZ5M01oWmc2WllwMXFwcWRvdkhKZ2lpU2JmMi85cmlEbW1uaHRJdi9iMmtzbDE)Starlight
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-midnight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=dUh3cXZ0dXFPVWlvS0EydFMzWWFVdUduSURuNUhkK2pOZnE1VnZHT1ZSWHZLN2M3ZGNkS1NNWStVS2JNbmRmN2RobUVKV0dYZHJrUWxHbk5lRDZzaFVtVDkrYjh1cXBHVkh2dmRxL2dHRkdWMlBxNmwzMWhWdTZFNk13V1NZRHk)Midnight
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    16GB Unified Memory  
    256GB SSD Storage  footnote  ¹
    * 16-core Neural Engine
    * 15.3-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/store-feature-mac-btr-appleintelligence?wid=40&hei=40&fmt=png-alpha&.v=L2k0UWhTOS9yRDNpVUx5cHpsQmZ2QVFRNnlzRVVmQlhLLy9yOFBWUWhWUjRCdVRQdmxzSUpzNHlRVmliMmtQYlEwTnpsQWxmZlludFcyRm4wOWcxaTFtcXBPRUNsRkE2bCtPZmcxeHdTcnpHbDRFMFhQRUtYQ25zNWRCb3liYzQ)Apple Intelligence Footnote ∆
    $1,199.00
    $99.91/mo.per month for 12 mo.monthsFootnote *
    Get 3% Daily Cash with Apple Card
    ### Apple Trade In
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/tradein-icon-currency?wid=50&hei=50&fmt=png-alpha&.v=SHZ4TEtxLzFWZTROQ21qbUU5QnN6WFpLVyt6cGFLME5kV2hjK3Q2TmdSdXpCNjVQRjg4dWNzbjlzU2FEVW1Qb25hT3pZVkRHQkc0Sk1OWE5mMzlJR3NzS0hkb3pGTXJoWThTNkIrNkxEaUU)
    ### Apple Trade In
    Get credit toward a new Mac when you trade in your eligible computer. Or recycle it for free.Footnote◊◊
    How does trade-in work?
    Get started with trade inM4 chip with 10-core CPU 10-core GPU Processor 256GB Storage 16GB memory
    SelectM4 chip with 10-core CPU 10-core GPU Processor 256GB Storage 16GB memory 
    ### Need a moment?
    Keep all your selections by saving this device to Your Saves, then come back anytime and pick up right where you left off.
    Save for later
    ![15-inch MacBook Air, top open, Liquid Retina display, rounded corners, raised feet, Sky Blue color](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-skyblue-select-202503?wid=904&hei=840&fmt=jpeg&qlt=90&.v=REV4NmZ6SUhUbzJzVXZrcXZ3UGg2NjQ1bzN1SitYTU83Mm9wbk1xa1lWNC9UNzNvY2N5NXJTTDQ2YkVYYmVXakJkRlpCNVhYU3AwTldRQldlSnpRa0JGbFFCaXFWTk5QRkxaWFZ6TExmVXM)Select a finish M4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 16GB memorySky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-skyblue-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=REV4NmZ6SUhUbzJzVXZrcXZ3UGg2N0ZUK08zZUpDWjVkdThxRjFOeXpiZ1loZk12ZmYvUVdUWXN6enVNb1ZMV1lJb0k2dU9zNW9xMXJqZDBvM2Qyb0xTNE1pQi9QM1JHSG9jeVpzRWk0Y3FOZkl3UzBsTFI3bGFSRmthOHVTSW4)Sky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-silver-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=YmVzOXFnNklRUk9jbTVLMFNybEcyRTd6N3NGS052Vk8vV1djRWEyczF6enBnUExpanZJKzYrcTVScDVOQkk1S0NBd3lOUFpnTTVCeDVDYzlNNEhMcFFPS3NYaFB3eWhNUXRVbVdTUWppalVKbndhbjJyQUQ2Q0NQdmRNNWZJWis)Silver
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-starlight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=TkljV1hUT21PWENUbGQ5c24rTjNVYnRZWE8yTitQUnBsaDIyYUxJdEFnMjRtQkh1eFY2UEs1YzJiZnNMZGs4LzBnYTc5dzVwRHNGRnI0TUhlVTZ5M01oWmc2WllwMXFwcWRvdkhKZ2lpU2JmMi85cmlEbW1uaHRJdi9iMmtzbDE)Starlight
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-midnight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=dUh3cXZ0dXFPVWlvS0EydFMzWWFVdUduSURuNUhkK2pOZnE1VnZHT1ZSWHZLN2M3ZGNkS1NNWStVS2JNbmRmN2RobUVKV0dYZHJrUWxHbk5lRDZzaFVtVDkrYjh1cXBHVkh2dmRxL2dHRkdWMlBxNmwzMWhWdTZFNk13V1NZRHk)Midnight
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    16GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    * 16-core Neural Engine
    * 15.3-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/store-feature-mac-btr-appleintelligence?wid=40&hei=40&fmt=png-alpha&.v=L2k0UWhTOS9yRDNpVUx5cHpsQmZ2QVFRNnlzRVVmQlhLLy9yOFBWUWhWUjRCdVRQdmxzSUpzNHlRVmliMmtQYlEwTnpsQWxmZlludFcyRm4wOWcxaTFtcXBPRUNsRkE2bCtPZmcxeHdTcnpHbDRFMFhQRUtYQ25zNWRCb3liYzQ)Apple Intelligence Footnote ∆
    $1,399.00
    $116.58/mo.per month for 12 mo.monthsFootnote *
    Get 3% Daily Cash with Apple Card
    ### Apple Trade In
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/tradein-icon-currency?wid=50&hei=50&fmt=png-alpha&.v=SHZ4TEtxLzFWZTROQ21qbUU5QnN6WFpLVyt6cGFLME5kV2hjK3Q2TmdSdXpCNjVQRjg4dWNzbjlzU2FEVW1Qb25hT3pZVkRHQkc0Sk1OWE5mMzlJR3NzS0hkb3pGTXJoWThTNkIrNkxEaUU)
    ### Apple Trade In
    Get credit toward a new Mac when you trade in your eligible computer. Or recycle it for free.Footnote◊◊
    How does trade-in work?
    Get started with trade inM4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 16GB memory
    SelectM4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 16GB memory 
    ### Need a moment?
    Keep all your selections by saving this device to Your Saves, then come back anytime and pick up right where you left off.
    Save for later
    ![15-inch MacBook Air, top open, Liquid Retina display, rounded corners, raised feet, Sky Blue color](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-skyblue-select-202503?wid=904&hei=840&fmt=jpeg&qlt=90&.v=REV4NmZ6SUhUbzJzVXZrcXZ3UGg2NjQ1bzN1SitYTU83Mm9wbk1xa1lWNC9UNzNvY2N5NXJTTDQ2YkVYYmVXakJkRlpCNVhYU3AwTldRQldlSnpRa0JGbFFCaXFWTk5QRkxaWFZ6TExmVXM)Select a finish M4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 24gb memorySky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-skyblue-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=REV4NmZ6SUhUbzJzVXZrcXZ3UGg2N0ZUK08zZUpDWjVkdThxRjFOeXpiZ1loZk12ZmYvUVdUWXN6enVNb1ZMV1lJb0k2dU9zNW9xMXJqZDBvM2Qyb0xTNE1pQi9QM1JHSG9jeVpzRWk0Y3FOZkl3UzBsTFI3bGFSRmthOHVTSW4)Sky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-silver-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=YmVzOXFnNklRUk9jbTVLMFNybEcyRTd6N3NGS052Vk8vV1djRWEyczF6enBnUExpanZJKzYrcTVScDVOQkk1S0NBd3lOUFpnTTVCeDVDYzlNNEhMcFFPS3NYaFB3eWhNUXRVbVdTUWppalVKbndhbjJyQUQ2Q0NQdmRNNWZJWis)Silver
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-starlight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=TkljV1hUT21PWENUbGQ5c24rTjNVYnRZWE8yTitQUnBsaDIyYUxJdEFnMjRtQkh1eFY2UEs1YzJiZnNMZGs4LzBnYTc5dzVwRHNGRnI0TUhlVTZ5M01oWmc2WllwMXFwcWRvdkhKZ2lpU2JmMi85cmlEbW1uaHRJdi9iMmtzbDE)Starlight
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-midnight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=dUh3cXZ0dXFPVWlvS0EydFMzWWFVdUduSURuNUhkK2pOZnE1VnZHT1ZSWHZLN2M3ZGNkS1NNWStVS2JNbmRmN2RobUVKV0dYZHJrUWxHbk5lRDZzaFVtVDkrYjh1cXBHVkh2dmRxL2dHRkdWMlBxNmwzMWhWdTZFNk13V1NZRHk)Midnight
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    24GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    * 16-core Neural Engine
    * 15.3-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/store-feature-mac-btr-appleintelligence?wid=40&hei=40&fmt=png-alpha&.v=L2k0UWhTOS9yRDNpVUx5cHpsQmZ2QVFRNnlzRVVmQlhLLy9yOFBWUWhWUjRCdVRQdmxzSUpzNHlRVmliMmtQYlEwTnpsQWxmZlludFcyRm4wOWcxaTFtcXBPRUNsRkE2bCtPZmcxeHdTcnpHbDRFMFhQRUtYQ25zNWRCb3liYzQ)Apple Intelligence Footnote ∆
    $1,599.00
    $133.25/mo.per month for 12 mo.monthsFootnote *
    Get 3% Daily Cash with Apple Card
    ### Apple Trade In
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/tradein-icon-currency?wid=50&hei=50&fmt=png-alpha&.v=SHZ4TEtxLzFWZTROQ21qbUU5QnN6WFpLVyt6cGFLME5kV2hjK3Q2TmdSdXpCNjVQRjg4dWNzbjlzU2FEVW1Qb25hT3pZVkRHQkc0Sk1OWE5mMzlJR3NzS0hkb3pGTXJoWThTNkIrNkxEaUU)
    ### Apple Trade In
    Get credit toward a new Mac when you trade in your eligible computer. Or recycle it for free.Footnote◊◊
    How does trade-in work?
    Get started with trade inM4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 24gb memory
    SelectM4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 24gb memory 
    ### Need a moment?
    Keep all your selections by saving this device to Your Saves, then come back anytime and pick up right where you left off.
    Save for later
    Have questions about buying a Mac?  
    Chat with a Mac Specialist(Opens in a new window)
    Select a finish M4 chip with 10-core CPU 8-core GPU Processor 256GB Storage 16GB memory Sky Blue 
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-skyblue-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=M2RyY09CWXlTQUp1KzEveHR6VXNxckZUK08zZUpDWjVkdThxRjFOeXpiZ1loZk12ZmYvUVdUWXN6enVNb1ZMV1lJb0k2dU9zNW9xMXJqZDBvM2Qyb0xTNE1pQi9QM1JHSG9jeVpzRWk0Y3BTeVdNUUpkdUNVQmFQWUtkeTlaakc) Sky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-silver-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=cCtGdk55eHp5QXJpcHVMMTl2dWZRVTd6N3NGS052Vk8vV1djRWEyczF6enBnUExpanZJKzYrcTVScDVOQkk1S0NBd3lOUFpnTTVCeDVDYzlNNEhMcGVQdVlGd3VrOGlVYkpnMGk1RGpNTVg0TjVYSTlBSTJEWFRmd2Q1ejdzcUw) Silver
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-starlight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=U3RnbTBEaGs0WGRSa05FY2NWZjZNYnRZWE8yTitQUnBsaDIyYUxJdEFnMjRtQkh1eFY2UEs1YzJiZnNMZGs4LzBnYTc5dzVwRHNGRnI0TUhlVTZ5M01oWmc2WllwMXFwcWRvdkhKZ2lpU2F1dFJrcTkxU3lYM0RQR3RSN3V2bGs) Starlight
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-midnight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=MWFRUnJZcWcrd0xyc0tIOUdIbnJlK0duSURuNUhkK2pOZnE1VnZHT1ZSWHZLN2M3ZGNkS1NNWStVS2JNbmRmN2RobUVKV0dYZHJrUWxHbk5lRDZzaFVtVDkrYjh1cXBHVkh2dmRxL2dHRkZFYVdKMW9LeXZTaXRrdkpJWktJcng) Midnight
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    8-Core GPU  
    16GB Unified Memory  
    256GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 8-core GPU  
    256GB Storage
    * 16-core Neural Engine
    * 13.6-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 30W USB-C Power Adapter
    $999.00
    $83.25/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    8-Core GPU  
    16GB Unified Memory  
    256GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 8-core GPU  
    256GB Storage
    * 16-core Neural Engine
    * 13.6-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 30W USB-C Power Adapter
    $999.00
    $83.25/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    8-Core GPU  
    16GB Unified Memory  
    256GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 8-core GPU  
    256GB Storage
    * 16-core Neural Engine
    * 13.6-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 30W USB-C Power Adapter
    $999.00
    $83.25/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    8-Core GPU  
    16GB Unified Memory  
    256GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 8-core GPU  
    256GB Storage
    * 16-core Neural Engine
    * 13.6-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 30W USB-C Power Adapter
    $999.00
    $83.25/mo.per month for 12 mo.monthsFootnote *
    Select Select
    Select a finish M4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 16GB memory Sky Blue 
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-skyblue-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=M2RyY09CWXlTQUp1KzEveHR6VXNxckZUK08zZUpDWjVkdThxRjFOeXpiZ1loZk12ZmYvUVdUWXN6enVNb1ZMV1lJb0k2dU9zNW9xMXJqZDBvM2Qyb0xTNE1pQi9QM1JHSG9jeVpzRWk0Y3BTeVdNUUpkdUNVQmFQWUtkeTlaakc) Sky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-silver-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=cCtGdk55eHp5QXJpcHVMMTl2dWZRVTd6N3NGS052Vk8vV1djRWEyczF6enBnUExpanZJKzYrcTVScDVOQkk1S0NBd3lOUFpnTTVCeDVDYzlNNEhMcGVQdVlGd3VrOGlVYkpnMGk1RGpNTVg0TjVYSTlBSTJEWFRmd2Q1ejdzcUw) Silver
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-starlight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=U3RnbTBEaGs0WGRSa05FY2NWZjZNYnRZWE8yTitQUnBsaDIyYUxJdEFnMjRtQkh1eFY2UEs1YzJiZnNMZGs4LzBnYTc5dzVwRHNGRnI0TUhlVTZ5M01oWmc2WllwMXFwcWRvdkhKZ2lpU2F1dFJrcTkxU3lYM0RQR3RSN3V2bGs) Starlight
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-midnight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=MWFRUnJZcWcrd0xyc0tIOUdIbnJlK0duSURuNUhkK2pOZnE1VnZHT1ZSWHZLN2M3ZGNkS1NNWStVS2JNbmRmN2RobUVKV0dYZHJrUWxHbk5lRDZzaFVtVDkrYjh1cXBHVkh2dmRxL2dHRkZFYVdKMW9LeXZTaXRrdkpJWktJcng) Midnight
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    16GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 13.6-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,199.00
    $99.91/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    16GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 13.6-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,199.00
    $99.91/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    16GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 13.6-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,199.00
    $99.91/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    16GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 13.6-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,199.00
    $99.91/mo.per month for 12 mo.monthsFootnote *
    Select Select
    Select a finish M4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 24gb memory Sky Blue 
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-skyblue-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=M2RyY09CWXlTQUp1KzEveHR6VXNxckZUK08zZUpDWjVkdThxRjFOeXpiZ1loZk12ZmYvUVdUWXN6enVNb1ZMV1lJb0k2dU9zNW9xMXJqZDBvM2Qyb0xTNE1pQi9QM1JHSG9jeVpzRWk0Y3BTeVdNUUpkdUNVQmFQWUtkeTlaakc) Sky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-silver-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=cCtGdk55eHp5QXJpcHVMMTl2dWZRVTd6N3NGS052Vk8vV1djRWEyczF6enBnUExpanZJKzYrcTVScDVOQkk1S0NBd3lOUFpnTTVCeDVDYzlNNEhMcGVQdVlGd3VrOGlVYkpnMGk1RGpNTVg0TjVYSTlBSTJEWFRmd2Q1ejdzcUw) Silver
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-starlight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=U3RnbTBEaGs0WGRSa05FY2NWZjZNYnRZWE8yTitQUnBsaDIyYUxJdEFnMjRtQkh1eFY2UEs1YzJiZnNMZGs4LzBnYTc5dzVwRHNGRnI0TUhlVTZ5M01oWmc2WllwMXFwcWRvdkhKZ2lpU2F1dFJrcTkxU3lYM0RQR3RSN3V2bGs) Starlight
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-midnight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=MWFRUnJZcWcrd0xyc0tIOUdIbnJlK0duSURuNUhkK2pOZnE1VnZHT1ZSWHZLN2M3ZGNkS1NNWStVS2JNbmRmN2RobUVKV0dYZHJrUWxHbk5lRDZzaFVtVDkrYjh1cXBHVkh2dmRxL2dHRkZFYVdKMW9LeXZTaXRrdkpJWktJcng) Midnight
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    24GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 13.6-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,399.00
    $116.58/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    24GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 13.6-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,399.00
    $116.58/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    24GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 13.6-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,399.00
    $116.58/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    24GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 13.6-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,399.00
    $116.58/mo.per month for 12 mo.monthsFootnote *
    Select Select
    Select a finish M4 chip with 10-core CPU 10-core GPU Processor 256GB Storage 16GB memory Sky Blue 
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-skyblue-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=REV4NmZ6SUhUbzJzVXZrcXZ3UGg2N0ZUK08zZUpDWjVkdThxRjFOeXpiZ1loZk12ZmYvUVdUWXN6enVNb1ZMV1lJb0k2dU9zNW9xMXJqZDBvM2Qyb0xTNE1pQi9QM1JHSG9jeVpzRWk0Y3FOZkl3UzBsTFI3bGFSRmthOHVTSW4) Sky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-silver-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=YmVzOXFnNklRUk9jbTVLMFNybEcyRTd6N3NGS052Vk8vV1djRWEyczF6enBnUExpanZJKzYrcTVScDVOQkk1S0NBd3lOUFpnTTVCeDVDYzlNNEhMcFFPS3NYaFB3eWhNUXRVbVdTUWppalVKbndhbjJyQUQ2Q0NQdmRNNWZJWis) Silver
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-starlight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=TkljV1hUT21PWENUbGQ5c24rTjNVYnRZWE8yTitQUnBsaDIyYUxJdEFnMjRtQkh1eFY2UEs1YzJiZnNMZGs4LzBnYTc5dzVwRHNGRnI0TUhlVTZ5M01oWmc2WllwMXFwcWRvdkhKZ2lpU2JmMi85cmlEbW1uaHRJdi9iMmtzbDE) Starlight
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-midnight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=dUh3cXZ0dXFPVWlvS0EydFMzWWFVdUduSURuNUhkK2pOZnE1VnZHT1ZSWHZLN2M3ZGNkS1NNWStVS2JNbmRmN2RobUVKV0dYZHJrUWxHbk5lRDZzaFVtVDkrYjh1cXBHVkh2dmRxL2dHRkdWMlBxNmwzMWhWdTZFNk13V1NZRHk) Midnight
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    16GB Unified Memory  
    256GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    256GB Storage
    * 16-core Neural Engine
    * 15.3-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,199.00
    $99.91/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    16GB Unified Memory  
    256GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    256GB Storage
    * 16-core Neural Engine
    * 15.3-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,199.00
    $99.91/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    16GB Unified Memory  
    256GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    256GB Storage
    * 16-core Neural Engine
    * 15.3-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,199.00
    $99.91/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    16GB Unified Memory  
    256GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    256GB Storage
    * 16-core Neural Engine
    * 15.3-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,199.00
    $99.91/mo.per month for 12 mo.monthsFootnote *
    Select Select
    Select a finish M4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 16GB memory Sky Blue 
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-skyblue-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=REV4NmZ6SUhUbzJzVXZrcXZ3UGg2N0ZUK08zZUpDWjVkdThxRjFOeXpiZ1loZk12ZmYvUVdUWXN6enVNb1ZMV1lJb0k2dU9zNW9xMXJqZDBvM2Qyb0xTNE1pQi9QM1JHSG9jeVpzRWk0Y3FOZkl3UzBsTFI3bGFSRmthOHVTSW4) Sky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-silver-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=YmVzOXFnNklRUk9jbTVLMFNybEcyRTd6N3NGS052Vk8vV1djRWEyczF6enBnUExpanZJKzYrcTVScDVOQkk1S0NBd3lOUFpnTTVCeDVDYzlNNEhMcFFPS3NYaFB3eWhNUXRVbVdTUWppalVKbndhbjJyQUQ2Q0NQdmRNNWZJWis) Silver
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-starlight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=TkljV1hUT21PWENUbGQ5c24rTjNVYnRZWE8yTitQUnBsaDIyYUxJdEFnMjRtQkh1eFY2UEs1YzJiZnNMZGs4LzBnYTc5dzVwRHNGRnI0TUhlVTZ5M01oWmc2WllwMXFwcWRvdkhKZ2lpU2JmMi85cmlEbW1uaHRJdi9iMmtzbDE) Starlight
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-midnight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=dUh3cXZ0dXFPVWlvS0EydFMzWWFVdUduSURuNUhkK2pOZnE1VnZHT1ZSWHZLN2M3ZGNkS1NNWStVS2JNbmRmN2RobUVKV0dYZHJrUWxHbk5lRDZzaFVtVDkrYjh1cXBHVkh2dmRxL2dHRkdWMlBxNmwzMWhWdTZFNk13V1NZRHk) Midnight
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    16GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 15.3-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,399.00
    $116.58/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    16GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 15.3-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,399.00
    $116.58/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    16GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 15.3-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,399.00
    $116.58/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    16GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 15.3-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,399.00
    $116.58/mo.per month for 12 mo.monthsFootnote *
    Select Select
    Select a finish M4 chip with 10-core CPU 10-core GPU Processor 512GB Storage 24gb memory Sky Blue 
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-skyblue-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=REV4NmZ6SUhUbzJzVXZrcXZ3UGg2N0ZUK08zZUpDWjVkdThxRjFOeXpiZ1loZk12ZmYvUVdUWXN6enVNb1ZMV1lJb0k2dU9zNW9xMXJqZDBvM2Qyb0xTNE1pQi9QM1JHSG9jeVpzRWk0Y3FOZkl3UzBsTFI3bGFSRmthOHVTSW4) Sky Blue
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-silver-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=YmVzOXFnNklRUk9jbTVLMFNybEcyRTd6N3NGS052Vk8vV1djRWEyczF6enBnUExpanZJKzYrcTVScDVOQkk1S0NBd3lOUFpnTTVCeDVDYzlNNEhMcFFPS3NYaFB3eWhNUXRVbVdTUWppalVKbndhbjJyQUQ2Q0NQdmRNNWZJWis) Silver
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-starlight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=TkljV1hUT21PWENUbGQ5c24rTjNVYnRZWE8yTitQUnBsaDIyYUxJdEFnMjRtQkh1eFY2UEs1YzJiZnNMZGs4LzBnYTc5dzVwRHNGRnI0TUhlVTZ5M01oWmc2WllwMXFwcWRvdkhKZ2lpU2JmMi85cmlEbW1uaHRJdi9iMmtzbDE) Starlight
    * ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba15-midnight-select-202503_SW_COLOR?wid=64&hei=64&fmt=jpeg&qlt=90&.v=dUh3cXZ0dXFPVWlvS0EydFMzWWFVdUduSURuNUhkK2pOZnE1VnZHT1ZSWHZLN2M3ZGNkS1NNWStVS2JNbmRmN2RobUVKV0dYZHJrUWxHbk5lRDZzaFVtVDkrYjh1cXBHVkh2dmRxL2dHRkdWMlBxNmwzMWhWdTZFNk13V1NZRHk) Midnight
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    24GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 15.3-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,599.00
    $133.25/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    24GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 15.3-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,599.00
    $133.25/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    24GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 15.3-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,599.00
    $133.25/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ### 
    ![Apple M4 Chip](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/m4-icon-202410?wid=102&hei=102&fmt=png-alpha&.v=UGhGV2V3SjNQb2Y0ekhXdW9ZOTZKVWNHU3dTWTNSMWRkQzZWaWhFVC9pL1RmMUFhaXkzWVI4b3hpbGUzQlh3YTJQNzFKaURON0I0cmR1S0VldldDalE)
    10-Core CPU  
    10-Core GPU  
    24GB Unified Memory  
    512GB SSD Storage  footnote  ¹
    ### M4 chip with 10-core CPU 10-core GPU  
    512GB Storage
    * 16-core Neural Engine
    * 15.3-inch Liquid Retina display with True Tone²
    * 12MP Center Stage camera
    * MagSafe 3 charging port
    * Two Thunderbolt 4 ports
    * Support for up to two external displays
    * Magic Keyboard with Touch ID
    * Force Touch trackpad
    * 35W Dual USB-C Port Compact Power Adapter
    $1,599.00
    $133.25/mo.per month for 12 mo.monthsFootnote *
    Select Select
    ##  Students and educators — save on a new Mac for college. 
    Get special pricing in the Education Store. 
    Shop now Students and educators — save on a new Mac for college. 
    ## What to consider when choosing your MacBook Air.
    Configure your laptop on the next step.
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mac-decision-m4-202410?wid=72&hei=72&fmt=png-alpha&.v=ZU5tSzV0VXNSYWF3bjJjbXpjdjJTeGg5ajdjbjdyK2QreDZwU29KUzEvYncybXQ2d2tyVWllaklsTldveFJyUkxkN2NaQ1pCS3o1WWUwL2tiWDE4OGZ0Yk5ndEJzcThwdHZ4TTVNWXpXZEE)
    ### Apple M4 chip
    M4 brings immense performance and capability, so you can blaze through everyday activities, multitask across apps and video calls, and handle elaborate content in creative apps and games. And with a faster Neural Engine, AI features within your apps fly.
    * Effortlessly run multiple apps
    * Edit thousands of photos, edit 4K video
    * Configure with up to 32GB unified memory
    * Supports up to two external displays
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/memory-icon-mac-202306?wid=88&hei=72&fmt=png-alpha&.v=YW1YQjRoQVdNdUdzdFNiR3JVcGZEQlpzenhKeTdKakdKU0dNdUljUmFnWHcybXQ2d2tyVWllaklsTldveFJyUkRPOXpDM2dTTW1BQVBDN1JlUTJxV1I1Nmx4SFNuUmF1YWhINm9DNTVrQkU)
    ### Unified Memory
    Faster and more efficient than traditional RAM, unified memory is integrated within Apple silicon so apps can quickly share data between the CPU, GPU, and Neural Engine.
    * Run multiple apps at once while performance remains fast and responsive
    * Add memory to run more apps simultaneously for faster, more fluid multitasking
    * MacBook Air can be configured with up to 32GB of memory
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/storage-icon-mac-202402?wid=90&hei=64&fmt=png-alpha&.v=UFpvRldxMFl5UDF0L1p5NjNra0dyR01acTBKMHVma01wRmphR1RpUkRqakxCZURPdllYOG8wQlc1OWdPczUrc2t0Mlk3VWtNd09KZVpySDVWby9nWGk2SjNVMUNVOWl4MVJGRUhsbFhML3c)
    ### Storage
    Solid-state drive (SSD) storage is the amount of space your Mac has for your documents, photos, music, videos, and other files.
    * Delivers exceptional performance and speed when you start up, launch apps, open files, and browse libraries
    * MacBook Air can be configured with up to 2TB of storage
    Have questions about buying a Mac? Chat with a Specialist(Opens in a new window)
    ## What’s in the Box
    * ![13-inch MacBook Air, top exterior, rounded corners, Apple logo centred, Sky Blue color](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-skyblue-witb-202503?wid=1096&hei=784&fmt=p-jpg&qlt=95&.v=d0xaZ3RhMTlBdXhXYXVJZnE2OEdxVUJ5ZVdoWWgzZFhvRmhMTjRuSlRMN2xFWFpkMzl1ckRtcktGc0VDM3VhUFMrR3RSUk9nckZNSURSVTdTTTRtUVhjaVBLd25LMmVWT0psdmg3ZTVaOVE)
    13-inch MacBook Air
    * ![USB-C to MagSafe charging cable, white USB-C connector, Sky Blue woven cable, Sky Blue MagSafe connector](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-cable-witb-202503?wid=192&hei=784&fmt=p-jpg&qlt=95&.v=ZnNyVXR1UmVTdzdKaFhFK3cvYmE0K3F2VUV5ek8zdVpTQmw1NU9vYzB4TE44TmxvS2tyTEFTd045dHRZV2FrQkNwdjgxNzdjcnlUVy83S0w5bWJ2TnVJNGQ5MEZOeVdmTnZvNzhTa1FxL0U)
    USB-C to MagSafe 3 Cable (2 m)
    * ![Power adapter, square, rounded corners, white](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mba13-adapter-witb-202503?wid=544&hei=784&fmt=p-jpg&qlt=95&.v=VHNMeENacDhNYnFzUjk5cGlhL2xuMXZETW5hUVRHWnRwZG9PY29RTldJUGU5K1krb0dpUG56VFN2T1VFRGlMcUtYeVk3dW9BYmZDa2xxQitSOWhMaHNQc2N4ZDBtYlFrUWxFdG54V2U0aTA)
    USB-C Power Adapter
    ##  Compare Mac models.
    Choose the best Mac laptop or desktop for you (opens in a new window)
    ![Mac models: 14-inch MacBook Pro, 16-inch MacBook Pro, iMac, Mac mini, Mac Studio, 15-inch MacBook Air, 13-inch MacBook Air, Mac Pro](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/mac-compare-202510?wid=1920&hei=710&fmt=png-alpha&.v=MHNaZys3eUhaejdmdnZtZjkrdWc0UnRacExUUU1yNSsybXc1aDhFTzMzWFl4OW5PQzZsaklFc2RhVndzZjZSekpJa1BYNE5MN2ZlMFNEU2FTNVBwc2o2endqbVZRb1oxSGptWGJqQnFCL0k)
    ![](https://store.storeimages.cdn-apple.com/1/as-images.apple.com/is/applecare-hero-bb-201706_GEO_US?wid=304&hei=304&fmt=jpeg&qlt=90&.v=ODdXYTlEUG9RdFhWUGFCckgvR3ZkZEswV0pybzRVYnRxdmsrelhPWTJub0wyQlgrV2pqTGxWNitVNUxyZHdxcDE1UUxLT2t0cW42N3FvQzVqaGhrVVpJaXMwN2lnNUN4aE5EVnBMb0RwSXQrYWpGdS9XeFgvbS9ITnNYOEhYaG4)
    ##  Mac. Handled with AppleCare. 
    AppleCare offers one-stop support and service for all of your Apple products — from the people who know them best. Get easy, fast repairs for accidents like drops and spills. And 24/7 priority care with just a chat, call, or tap. Footnote °
    You can enjoy individual coverage for Mac with AppleCare+. Or you can cover multiple products with AppleCare One. No matter which plan you choose, you’ll get peace of mind for the products you love. Footnote °°
    Learn more(opens in a new window)
    ##  Shopping for your business? 
    See how Apple at Work can help. 
    Learn more Shopping for your business?
    ## Footer
    ### footnotes
    ∆ Apple Intelligence is available in beta on all Mac models with M1 and later, with Siri and device language set to Chinese (Simplified), English (Australia, Canada, India, Ireland, New Zealand, Singapore, South Africa, UK or US), French, German, Italian, Japanese, Korean, Portuguese (Brazil) or Spanish, as a macOS Sequoia update, with more languages coming over the course of the year, including Vietnamese. Some features may not be available in all regions or languages.
    * Financing available to qualified customers, subject to credit approval and credit limit, and requires you to select Apple Card Monthly Installments (ACMI) as your payment type at checkout at Apple. Financing terms vary by product. Taxes and shipping on items purchased using ACMI are subject to your card’s variable APR, not the ACMI 0% APR. ACMI is not available for purchases made at special storefronts or when using such special discounts in-store at Apple, except ACMI is available at the Education storefront and with the Education discount. The last month’s payment for each product will be the product’s purchase price, less all other payments at the monthly payment amount. ACMI is subject to change at any time for any reason, including but not limited to installment term lengths and eligible products. See the Apple Card Customer Agreement (Opens in a new window) for more information about ACMI. 
    ◊ Apple Card Monthly Installments (ACMI) is a 0% APR payment option that is only available if you select it at checkout in the U.S. for eligible products purchased at Apple Store locations, apple.com (Opens in a new window), the Apple Store app, or by calling 1-800-MY-APPLE, and is subject to credit approval and credit limit. See here (Opens in a new window) for more information about eligible products. Existing customers: See your Customer Agreement for your variable APR. As of November 1, 2025, the variable APR on new Apple Card accounts ranges from 17.74% to 27.99%. You must elect to use ACMI at checkout. If you buy an ACMI-eligible product with a one-time payment on Apple Card at checkout, that purchase is subject to your Apple Card’s variable APR, not the ACMI 0% APR. Taxes and shipping on items purchased using ACMI are subject to your Apple Card’s variable APR, not the ACMI 0% APR. In order to buy an iPhone with ACMI, you must select one of the following carriers (prepaid carrier plans are not supported): AT&T, Boost Mobile, T-Mobile, or Verizon. An iPhone purchased with ACMI is always unlocked, so you can switch carriers at any time, subject to your carrier’s terms. ACMI is not available for purchases made at the following special storefronts or when using these discounts in-store at Apple: Apple Employee Purchase Plan; participating corporate Employee Purchase Programs; Apple at Work for small businesses; Government and Veterans and Military Purchase Programs; or on refurbished devices. The last month’s payment for each product will be the product’s purchase price, less all other payments at the monthly payment amount. ACMI is subject to change at any time for any reason, including but not limited to installment term lengths and eligible products. See the Apple Card Customer Agreement (Opens in a new window) for more information about ACMI.
    To access and use all Apple Card features and products available only to Apple Card users, you must add Apple Card to Wallet on an iPhone or iPad that supports and has the latest version of iOS or iPadOS. Apple Card is subject to credit approval, available only for qualifying applicants in the United States, and issued by Goldman Sachs Bank USA, Salt Lake City Branch.
    Apple Payments Services LLC, a subsidiary of Apple Inc., is a service provider of Goldman Sachs Bank USA for Apple Card and Savings accounts. Neither Apple Inc. nor Apple Payments Services LLC is a bank.
    All communications from Apple and Goldman Sachs Bank USA about Apple Card (including transactional and marketing communications) and customer service support are available in English. Certain communications about Apple Card can be viewed in another language depending on your device language settings. If you reside in the U.S. Virgin Islands, American Samoa, Guam, Northern Mariana Islands, or U.S. Minor Outlying Islands, please call Goldman Sachs at 877-255-5923 with questions about Apple Card.
    ◊◊ Trade-in values will vary based on the condition, year, and configuration of your eligible trade-in device. Not all devices are eligible for credit. You must be at least the age of majority to be eligible to trade in for credit or for an Apple Gift Card. Trade-in value may be applied toward qualifying new device purchase, or added to an Apple Gift Card. Actual value awarded is based on receipt of a qualifying device matching the description provided when estimate was made. Sales tax may be assessed on full value of a new device purchase. In-store trade-in requires presentation of a valid photo ID (local law may require saving this information). Offer may not be available in all stores, and may vary between in-store and online trade-in. Some stores may have additional requirements. Apple or its trade-in partners reserve the right to refuse, cancel, or limit quantity of any trade-in transaction for any reason. More details are available from Apple’s trade-in partner for trade-in and recycling of eligible devices. Restrictions and limitations may apply.
    1. 1GB = 1 billion bytes and 1TB = 1 trillion bytes; actual formatted capacity less.
    2. Screen size is measured diagonally. The displays on the 13‑inch and 15‑inch MacBook Air have rounded corners at the top. When measured as a standard rectangular shape, the screens are 13.6 inches and 15.3 inches diagonally (actual viewable area is less).
    ° Local telephone fees may apply. Telephone numbers and hours of operation may vary and are subject to change.
    °° Service coverage is available only for covered devices and their original included accessories for protection against (i) defects in materials or workmanship, (ii) batteries that retain less than 80% of their original capacity, and (iii) unlimited incidents of accidental damage. Accidental Damage means physical damage from handling due to unexpected and unintentional events. If an iPad is covered in your plan, one compatible Apple Pencil and one compatible Apple-branded iPad keyboard used with your iPad are also covered. AppleCare One is subject to eligibility rules; additions to plan may require inspection and a diagnostic check: Devices must be less than 4 years old and headphones must be less than 1 year old, and only devices in your Apple Account can be covered under AppleCare One. Limit of one (1) Apple Vision Pro can be covered under AppleCare One at a time. Replacement equipment that Apple provides as part of the repair or replacement service may contain new or previously used genuine Apple parts that have been tested and pass Apple functional requirements. There are no service fees for mechanical failures. Each incident of accidental damage protection is subject to a service fee plus applicable tax. If you have a plan with theft and loss coverage, each incident of theft or loss is subject to a deductible plus applicable tax. For complete details, see the AppleCare One Terms and Conditions(Opens in a new window) and the Theft and Loss Insurance Documentation(Opens in a new window) applicable to your state. For AppleCare+, see Terms and Conditions(Opens in a new window) and the Theft and Loss Insurance Documentation(Opens in a new window) applicable to your state.
    We approximate your location from your internet IP address by matching it to a geographic region or from the location entered during your previous visit to Apple.
    Apple
    1. Mac
    2. MacBook Air
    3. Buy MacBook Air
    ###  Shop and Learn Shop and Learn
    * Store 
    * Mac 
    * iPad 
    * iPhone 
    * Watch 
    * Vision 
    * AirPods 
    * TV & Home 
    * AirTag 
    * Accessories 
    * Gift Cards 
    ###  Apple Wallet Apple Wallet
    * Wallet 
    * Apple Card 
    * Apple Pay 
    * Apple Cash 
    ###  Account Account
    * Manage Your Apple Account 
    * Apple Store Account 
    * iCloud.com 
    ###  Entertainment Entertainment
    * Apple One 
    * Apple TV 
    * Apple Music 
    * Apple Arcade 
    * Apple Fitness+ 
    * Apple News+ 
    * Apple Podcasts 
    * Apple Books 
    * App Store 
    ###  Apple Store Apple Store
    * Find a Store 
    * Genius Bar 
    * Today at Apple 
    * Apple Camp 
    * Apple Store App 
    * Certified Refurbished 
    * Apple Trade In 
    * Financing 
    * Carrier Deals at Apple 
    * Order Status 
    * Shopping Help 
    ###  For Business For Business
    * Apple and Business 
    * Shop for Business 
    ###  For Education For Education
    * Apple and Education 
    * Shop for K-12 
    * Shop for College 
    ###  For Healthcare For Healthcare
    * Apple in Healthcare 
    * Health on Apple Watch 
    * Health Records on iPhone 
    ###  For Government For Government
    * Shop for Government 
    * Shop for Veterans and Military 
    ###  Apple Values Apple Values
    * Accessibility 
    * Education 
    * Environment 
    * Inclusion and Diversity 
    * Privacy 
    * Racial Equity and Justice 
    * Supply Chain 
    ###  About Apple About Apple
    * Newsroom 
    * Apple Leadership 
    * Career Opportunities 
    * Investors 
    * Ethics & Compliance 
    * Events 
    * Contact Apple 
    More ways to shop: Find an Apple Store or other retailer near you. Or call 1‑800‑MY‑APPLE.
    United States 
    Copyright © 2025 Apple Inc. All rights reserved. 
    * Privacy Policy
    * Terms of Use
    * Sales and Refunds
    * Legal
    * Site Map
    ![](/shop/dc)![](/shop/dc)
    </webpage_content>"""
    
    # print("=== Test Normal Mode API for Webpage Content Extraction ===\n")
    # resp = requests.post(
    #     f"{BASE_URL}/v1/chat/completions",
    #     json={
    #         "mode": "normal",
    #         "messages": [
    #             {"role": "system", "content": system_message},
    #             {"role": "user", "content": prompt}
    #         ],
    #         "max_tokens": 1024,
    #     }
    # )
    # result = resp.json()
    # print(f"Content: {result['choices'][0]['message']['content']}")
    # print(f"Usage: {result['usage']}")


    print("=== Test LatentMAS API for Webpage Content Extraction ===\n")
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "mode": "latent",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            # latent_steps not specified - will use dynamic stopping
            "latent_steps": 5,
            "latent_space_realign": latent_space_realign,
            "debug_max_tokens": None,
            "debug_continuation_prompt": None,
            "latent_only": latent_only,
        }
    )
    result = resp.json()
    session_id = result["session_id"]
    print(f"Session ID: {result['session_id']}")
    print(f"Content: {result['choices'][0]['message']['content']}")
    print(f"Usage: {result['usage']}")
    print(f"KV cache shape: {result['usage'].get('kv_cache_shape')}")
    
    # Now use the session to generate final text
    print("\n[Text Generation] Using cached KV from dynamic latent")
    resp2 = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "mode": "text",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Please summarize what is in the KV cache:"}
            ],
            "session_id": session_id,
            "max_tokens": 1024,
        }
    )
    result2 = resp2.json()
    print(f"Final answer: {result2['choices'][0]['message']['content']}")
    
    return resp.status_code == 200

    
if __name__ == "__main__":
    print("LatentMAS API Test\n")
    
    try:
        # Run tests
        # test_health()
        # test_normal_mode()
        # test_latent_sequential()
        # test_latent_dynamic()
        # test_session_management()
        test_kv_injection_ablation()  # NEW: Verify KV injection is working
        # test_complex_task()
        
        print("\n=== All tests completed ===")
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server. Make sure the server is running:")
        print("  python server.py --model_name Qwen/Qwen2.5-7B-Instruct")
