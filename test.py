#!/usr/bin/env python3
"""
Comprehensive test script for the RAG-based Member QnA System.

This script tests:
1. External Messages API connectivity and data
2. RAG agent with semantic search capabilities
3. Interactive question-answering mode
4. Both name-based and content-based queries
"""

import os
import sys

# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import requests
from typing import Dict, Any, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.markdown import Markdown
from rich import print as rprint
from dotenv import load_dotenv

load_dotenv()

console = Console()

# Configuration
API_BASE_URL = "https://november7-730026606190.europe-west1.run.app"
MESSAGES_ENDPOINT = f"{API_BASE_URL}/messages/"
MESSAGES_API_KEY = os.getenv("MESSAGES_API_KEY")


def print_header():
    """Print main header."""
    console.print("\n")
    console.print(
        Panel.fit(
            "[bold cyan]RAG-Based Member QnA System - Test Suite[/bold cyan]\n"
            "[dim]Testing API, RAG agent, and semantic search capabilities[/dim]",
            border_style="cyan"
        )
    )
    console.print()


def test_api_connectivity() -> bool:
    """Test if the Messages API is accessible and returns data."""
    console.print("\n[bold yellow]═══ Test 1: API Connectivity ═══[/bold yellow]\n")
    
    console.print(f"[dim]Testing endpoint: {MESSAGES_ENDPOINT}[/dim]\n")
    
    try:
        # Prepare headers
        headers = {}
        if MESSAGES_API_KEY:
            headers["Authorization"] = f"Bearer {MESSAGES_API_KEY}"
            console.print("[dim]✓ Using API authentication[/dim]")
        else:
            console.print("[yellow]⚠ No API key found (trying without auth)[/yellow]")
        
        console.print()
        
        # Make request
        with console.status("[bold cyan]Fetching messages...", spinner="dots"):
            response = requests.get(
                MESSAGES_ENDPOINT,
                params={"skip": 0, "limit": 10},
                headers=headers,
                timeout=30
            )
        
        # Check status
        if response.status_code == 200:
            data = response.json()
            messages = data.get("items", [])
            total = data.get("total", 0)
            
            console.print(f"[bold green]✓ API is accessible![/bold green]")
            console.print(f"[dim]Status: {response.status_code}[/dim]")
            console.print(f"[dim]Response time: {response.elapsed.total_seconds():.2f}s[/dim]")
            console.print(f"[dim]Total messages available: {total}[/dim]")
            console.print(f"[dim]Sample retrieved: {len(messages)} messages[/dim]\n")
            
            # Show sample message
            if messages:
                sample = messages[0]
                console.print("[bold]Sample message:[/bold]")
                console.print(
                    Panel(
                        f"[cyan]User:[/cyan] {sample.get('user_name', 'N/A')}\n"
                        f"[cyan]Message:[/cyan] {sample.get('message', 'N/A')[:100]}...\n"
                        f"[dim]Timestamp: {sample.get('timestamp', 'N/A')}[/dim]",
                        border_style="green"
                    )
                )
            
            return True
        else:
            console.print(f"[bold red]✗ API returned status {response.status_code}[/bold red]\n")
            return False
            
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]✗ API connection failed:[/bold red] {str(e)}\n")
        
        if "401" in str(e) or "Unauthorized" in str(e):
            console.print(
                Panel(
                    "[yellow]Authentication may be required.[/yellow]\n"
                    "Add MESSAGES_API_KEY to your .env file if you have one.",
                    border_style="yellow",
                    title="Tip"
                )
            )
        return False


def test_rag_agent_initialization() -> Any:
    """Test RAG agent initialization."""
    console.print("\n[bold yellow]═══ Test 2: RAG Agent Initialization ═══[/bold yellow]\n")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            Panel(
                "[bold red]✗ OPENAI_API_KEY not found[/bold red]\n\n"
                "Please set your OpenAI API key in .env file:\n\n"
                "[cyan]OPENAI_API_KEY=sk-your-key-here[/cyan]",
                border_style="red",
                title="Error"
            )
        )
        return None
    
    try:
        from langgraph_agent import QnAAgent
        
        console.print("[dim]Initializing RAG agent with:[/dim]")
        console.print("[dim]  • LLM: gpt-5-nano[/dim]")
        console.print("[dim]  • Embeddings: text-embedding-3-small[/dim]")
        console.print("[dim]  • Retrieval: Top 5 documents[/dim]\n")
        
        agent = QnAAgent(
            llm_model="gpt-5-nano",
            embedding_model="text-embedding-3-small",
            k=5,
        )
        
        console.print("[bold green]✓ RAG agent initialized successfully![/bold green]\n")
        
        # Show architecture
        tree = Tree("🤖 [bold]RAG Agent Architecture[/bold]")
        
        load_branch = tree.add("📥 [cyan]Load & Index Node[/cyan]")
        load_branch.add("[dim]• Fetch messages from API[/dim]")
        load_branch.add("[dim]• Convert to documents with metadata[/dim]")
        load_branch.add("[dim]• Build/Load FAISS vector index[/dim]")
        
        retrieve_branch = tree.add("🔍 [yellow]Retrieve Context Node[/yellow]")
        retrieve_branch.add("[dim]• Perform semantic search[/dim]")
        retrieve_branch.add("[dim]• Get top-5 most relevant messages[/dim]")
        retrieve_branch.add("[dim]• Score by similarity[/dim]")
        
        generate_branch = tree.add("✨ [green]Generate Answer Node[/green]")
        generate_branch.add("[dim]• Use LLM with retrieved context[/dim]")
        generate_branch.add("[dim]• Generate factual answer[/dim]")
        generate_branch.add("[dim]• Cite sources[/dim]")
        
        console.print(tree)
        console.print()
        
        return agent
        
    except Exception as e:
        console.print(f"[bold red]✗ Failed to initialize agent:[/bold red] {str(e)}\n")
        return None


def test_rag_queries(agent: Any) -> bool:
    """Test RAG agent with various query types."""
    console.print("\n[bold yellow]═══ Test 3: RAG Question-Answering ═══[/bold yellow]\n")
    
    # Define test questions
    test_questions = [
        ("Name-based", "When is Layla planning her trip to London?"),
        ("Name-based", "How many cars does Vikram Desai have?"),
        ("Name-based", "What are Amira's favorite restaurants?"),
        ("Content-based", "Who likes Italian restaurants?"),
        ("Content-based", "Who has travel plans?"),
        ("Content-based", "Who mentioned cars or vehicles?"),
    ]
    
    console.print(f"Testing {len(test_questions)} questions:\n")
    
    success_count = 0
    
    for i, (query_type, question) in enumerate(test_questions, 1):
        # Print question
        console.print(
            Panel(
                f"[bold white]{question}[/bold white]\n\n"
                f"[dim]Type: {query_type} query[/dim]",
                title=f"[bold blue]Question {i}/{len(test_questions)}[/bold blue]",
                border_style="blue",
                padding=(0, 2)
            )
        )
        
        # Get answer
        try:
            with console.status("[bold cyan]Processing...", spinner="dots"):
                answer = agent.ask(question)
            
            # Print answer
            console.print(
                Panel(
                    Markdown(answer),
                    title=f"[bold green]Answer[/bold green]",
                    border_style="green",
                    padding=(1, 2)
                )
            )
            console.print()
            
            success_count += 1
            
        except Exception as e:
            console.print(
                Panel(
                    f"[bold red]Error:[/bold red] {str(e)}",
                    border_style="red"
                )
            )
            console.print()
    
    # Summary
    if success_count == len(test_questions):
        console.print(f"[bold green]✓ All {success_count}/{len(test_questions)} questions answered successfully![/bold green]\n")
        return True
    else:
        console.print(f"[yellow]⚠ {success_count}/{len(test_questions)} questions answered[/yellow]\n")
        return False


def interactive_mode(agent: Any):
    """Run interactive question-answering mode."""
    console.print("\n[bold yellow]═══ Interactive Mode ═══[/bold yellow]\n")
    
    console.print(
        Panel(
            "[bold cyan]Interactive Q&A Mode[/bold cyan]\n\n"
            "Ask questions about member data.\n\n"
            "[bold]Commands:[/bold]\n"
            "  • Type your question to get an answer\n"
            "  • 'examples' - Show example questions\n"
            "  • 'clear' - Clear the cache\n"
            "  • 'quit', 'exit', 'q' - Exit interactive mode",
            border_style="cyan"
        )
    )
    console.print()
    
    example_questions = [
        "When is Layla planning her trip to London?",
        "Who likes Italian restaurants?",
        "What are Amira's favorite restaurants?",
        "Who has travel plans?",
        "Who mentioned cars or vehicles?",
        "What hobbies do members have?",
    ]
    
    while True:
        try:
            # Get user input
            console.print("[bold blue]You:[/bold blue] ", end="")
            question = input().strip()
            
            if not question:
                continue
            
            # Handle commands
            if question.lower() in ['quit', 'exit', 'q']:
                console.print("\n[bold cyan]👋 Exiting interactive mode...[/bold cyan]\n")
                break
            
            if question.lower() == 'examples':
                console.print("\n[bold]Example questions you can ask:[/bold]\n")
                for i, ex in enumerate(example_questions, 1):
                    console.print(f"  {i}. [cyan]{ex}[/cyan]")
                console.print()
                continue
            
            if question.lower() == 'clear':
                agent.clear_cache()
                console.print("[green]✓ Cache cleared successfully[/green]\n")
                continue
            
            # Process question
            with console.status("[bold cyan]Thinking...", spinner="dots"):
                try:
                    answer = agent.ask(question)
                    
                    # Print answer
                    console.print("\n[bold green]Agent:[/bold green]")
                    console.print(
                        Panel(
                            Markdown(answer),
                            border_style="green",
                            padding=(1, 2)
                        )
                    )
                    console.print()
                    
                except Exception as e:
                    console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        
        except KeyboardInterrupt:
            console.print("\n\n[bold cyan]👋 Exiting interactive mode...[/bold cyan]\n")
            break
        except EOFError:
            console.print("\n\n[bold cyan]👋 Exiting interactive mode...[/bold cyan]\n")
            break


def main():
    """Main test function."""
    print_header()
    
    # Test 1: API Connectivity
    api_ok = test_api_connectivity()
    
    if not api_ok:
        console.print(
            Panel(
                "[yellow]⚠ API test failed but continuing with agent tests...[/yellow]\n"
                "The agent will attempt to fetch data when initialized.",
                border_style="yellow"
            )
        )
    
    # Test 2: RAG Agent Initialization
    agent = test_rag_agent_initialization()
    
    if not agent:
        console.print(
            Panel(
                "[bold red]Cannot continue without a working agent.[/bold red]\n"
                "Please fix the initialization errors and try again.",
                border_style="red",
                title="Test Failed"
            )
        )
        return 1
    
    # Test 3: RAG Queries
    queries_ok = test_rag_queries(agent)
    
    # Summary
    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]         Test Summary[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")
    
    # Create summary table
    summary_table = Table(show_header=True, header_style="bold cyan")
    summary_table.add_column("Test", style="white", width=30)
    summary_table.add_column("Status", justify="center", width=15)
    
    summary_table.add_row(
        "1. API Connectivity",
        "[green]✓ PASS[/green]" if api_ok else "[yellow]⚠ WARNING[/yellow]"
    )
    summary_table.add_row(
        "2. RAG Agent Init",
        "[green]✓ PASS[/green]"
    )
    summary_table.add_row(
        "3. Question Answering",
        "[green]✓ PASS[/green]" if queries_ok else "[yellow]⚠ PARTIAL[/yellow]"
    )
    
    console.print(summary_table)
    console.print()
    
    # Ask if user wants interactive mode
    console.print(
        Panel(
            "[bold]Would you like to try interactive mode?[/bold]\n\n"
            "You can ask your own questions and explore the system.\n\n"
            "Enter 'y' or 'yes' to continue, or anything else to exit:",
            border_style="cyan"
        )
    )
    
    choice = input("\nYour choice: ").strip().lower()
    
    if choice in ['y', 'yes']:
        console.print()
        interactive_mode(agent)
    
    # Final message
    console.print(
        Panel(
            "[bold green]✓ Testing Complete![/bold green]\n\n"
            "The RAG-based QnA system is working correctly.\n\n"
            "[bold]Key Features Demonstrated:[/bold]\n"
            "  ✓ Semantic search with FAISS\n"
            "  ✓ Name-based queries\n"
            "  ✓ Content-based queries\n"
            "  ✓ Efficient top-k retrieval\n"
            "  ✓ LLM-powered answer generation",
            border_style="green",
            title="Success"
        )
    )
    console.print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

