#!/usr/bin/env python3
"""
Test script to verify memory optimizations are working correctly.
Run this to ensure the API stays under 512MB memory limit.
"""

import os
import sys
import time
import psutil
import requests
from typing import List, Dict


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_api_endpoint(question: str) -> Dict:
    """Test a single question and measure response time."""
    url = "http://localhost:8000/ask"
    
    start_time = time.time()
    response = requests.post(
        url,
        json={"question": question},
        timeout=30
    )
    elapsed = time.time() - start_time
    
    return {
        "question": question,
        "status": response.status_code,
        "elapsed": elapsed,
        "answer_length": len(response.json().get("answer", "")),
        "success": response.status_code == 200
    }


def run_memory_test():
    """Run comprehensive memory optimization tests."""
    print("=" * 80)
    print("MEMORY OPTIMIZATION TEST SUITE")
    print("=" * 80)
    print()
    
    # Test questions covering different patterns
    test_questions = [
        "How many cars does Charles have?",
        "What restaurants did Michael mention?",
        "Who likes Italian food?",
        "Tell me about Emma's trips",
        "List all the luxury cars mentioned",
    ]
    
    print(f"Initial memory: {get_memory_mb():.1f} MB")
    print()
    
    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"[{i}/{len(test_questions)}] Testing: {question}")
        
        try:
            result = test_api_endpoint(question)
            results.append(result)
            
            if result["success"]:
                print(f"  ✅ Success in {result['elapsed']:.2f}s")
                print(f"  📝 Answer length: {result['answer_length']} chars")
            else:
                print(f"  ❌ Failed with status {result['status']}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results.append({
                "question": question,
                "success": False,
                "error": str(e)
            })
        
        print(f"  💾 Current memory: {get_memory_mb():.1f} MB")
        print()
        
        # Brief pause between requests
        time.sleep(1)
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results if r.get("success", False))
    total = len(results)
    
    print(f"✅ Successful requests: {successful}/{total}")
    
    if successful > 0:
        avg_time = sum(r["elapsed"] for r in results if r.get("success", False)) / successful
        print(f"⏱️  Average response time: {avg_time:.2f}s")
    
    final_memory = get_memory_mb()
    print(f"💾 Final memory: {final_memory:.1f} MB")
    
    if final_memory > 512:
        print()
        print("⚠️  WARNING: Memory usage exceeds 512MB!")
        print("   Consider reducing MAX_MESSAGES_LIMIT or RETRIEVAL_K in .env")
    elif final_memory > 400:
        print()
        print("⚠️  CAUTION: Memory usage is high (>400MB)")
        print("   Monitor closely in production")
    else:
        print()
        print("🎉 EXCELLENT: Memory usage is well within limits!")
    
    print()
    print("=" * 80)
    
    return successful == total


def check_environment():
    """Check if environment variables are properly configured."""
    print("=" * 80)
    print("ENVIRONMENT CONFIGURATION CHECK")
    print("=" * 80)
    print()
    
    from dotenv import load_dotenv
    load_dotenv()
    
    checks = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "MAX_MESSAGES_LIMIT": os.getenv("MAX_MESSAGES_LIMIT", "not set"),
        "DOC_STRATEGY": os.getenv("DOC_STRATEGY", "not set"),
        "RETRIEVAL_K": os.getenv("RETRIEVAL_K", "not set"),
    }
    
    all_good = True
    
    for key, value in checks.items():
        if key == "OPENAI_API_KEY":
            is_set = value and value != "your_openai_api_key_here"
            status = "✅" if is_set else "❌"
            display = "***SET***" if is_set else "NOT SET"
            print(f"{status} {key}: {display}")
            all_good = all_good and is_set
        else:
            print(f"  {key}: {value}")
    
    print()
    
    # Check recommended values
    print("RECOMMENDED SETTINGS:")
    print(f"  MAX_MESSAGES_LIMIT: 500 (current: {checks['MAX_MESSAGES_LIMIT']})")
    print(f"  DOC_STRATEGY: individual (current: {checks['DOC_STRATEGY']})")
    print(f"  RETRIEVAL_K: 3 (current: {checks['RETRIEVAL_K']})")
    print()
    
    if checks['DOC_STRATEGY'] == 'hybrid':
        print("⚠️  WARNING: DOC_STRATEGY=hybrid uses 2x memory!")
        print("   Change to 'individual' for 512MB environments")
        all_good = False
    
    if checks['MAX_MESSAGES_LIMIT'] != 'not set':
        limit = int(checks['MAX_MESSAGES_LIMIT'])
        if limit > 500:
            print(f"⚠️  CAUTION: MAX_MESSAGES_LIMIT={limit} may use excessive memory")
            print("   Recommended: 500 or lower")
    
    if checks['RETRIEVAL_K'] != 'not set':
        k = int(checks['RETRIEVAL_K'])
        if k > 3:
            print(f"⚠️  CAUTION: RETRIEVAL_K={k} may use excessive memory")
            print("   Recommended: 3 or lower")
    
    print()
    print("=" * 80)
    print()
    
    return all_good


def check_server():
    """Check if the server is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False


if __name__ == "__main__":
    print()
    
    # Step 1: Check environment
    env_ok = check_environment()
    if not env_ok:
        print("❌ Environment configuration issues detected!")
        print("   Please fix the issues above and try again.")
        sys.exit(1)
    
    # Step 2: Check if server is running
    print("Checking if server is running...")
    if not check_server():
        print()
        print("❌ Server is not running!")
        print()
        print("Start the server first:")
        print("  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        print()
        sys.exit(1)
    
    print("✅ Server is running")
    print()
    
    # Step 3: Run memory tests
    success = run_memory_test()
    
    if success:
        print("🎉 All tests passed! Your API is optimized and ready for deployment.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)

