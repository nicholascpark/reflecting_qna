#!/bin/bash

# Performance Testing Script for Reflecting QnA API
# Usage: ./test_performance.sh <API_URL>
# Example: ./test_performance.sh https://your-app.onrender.com

API_URL="${1:-http://localhost:8000}"

echo "================================"
echo "Performance Testing Script"
echo "================================"
echo "API URL: $API_URL"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test 1: Health Check
echo -e "${BLUE}[Test 1]${NC} Health Check (should be instant)"
time curl -s "$API_URL/health" | jq '.'
echo ""

# Test 2: Warmup
echo -e "${BLUE}[Test 2]${NC} Warmup Endpoint (pre-loads FAISS index)"
time curl -s -X POST "$API_URL/warmup" | jq '.'
echo ""

# Test 3: First Question
echo -e "${BLUE}[Test 3]${NC} First Question (should be fast after warmup)"
time curl -s -X POST "$API_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the recent messages?"}' | jq '.'
echo ""

# Test 4: Second Question (cache should be warm)
echo -e "${BLUE}[Test 4]${NC} Second Question (should be faster)"
time curl -s -X POST "$API_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Who likes Italian restaurants?"}' | jq '.'
echo ""

# Test 5: Complex Question
echo -e "${BLUE}[Test 5]${NC} Complex Question (counting/aggregation)"
time curl -s -X POST "$API_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many cars are mentioned in the messages?"}' | jq '.'
echo ""

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Performance Testing Complete${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Expected response times:"
echo "  - Health check: < 0.1s"
echo "  - Warmup: 1-2s"
echo "  - Questions: 1-3s each"
echo ""
echo "If response times are much longer:"
echo "  1. Check if service is on free tier (cold starts)"
echo "  2. Check OpenAI API status"
echo "  3. Review logs in Render dashboard"
echo "  4. See PERFORMANCE.md for optimization tips"

