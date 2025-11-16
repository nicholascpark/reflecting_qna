#!/usr/bin/env python3
"""
Quick API status checker.

Simple script to verify the Messages API is accessible and working.
"""

import os
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

console = Console()

API_URL = os.getenv(
    "MESSAGES_API_URL",
    "https://november7-730026606190.europe-west1.run.app/messages/"
)
API_KEY = os.getenv("MESSAGES_API_KEY")


def check_api():
    """Quick check of the Messages API."""
    console.print("\n[bold cyan]🔍 Checking Messages API...[/bold cyan]\n")
    
    try:
        headers = {}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
            console.print("[dim]Using authentication[/dim]")
        else:
            console.print("[dim]No authentication (API key not set)[/dim]")
        
        response = requests.get(
            API_URL,
            params={"skip": 0, "limit": 1},
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            total = data.get("total", 0)
            
            console.print(
                Panel(
                    f"[bold green]✓ API is working![/bold green]\n\n"
                    f"URL: {API_URL}\n"
                    f"Total messages available: {total}\n"
                    f"Response time: {response.elapsed.total_seconds():.2f}s",
                    border_style="green",
                    title="[bold]Status: Healthy[/bold]"
                )
            )
            return True
        elif response.status_code == 401:
            console.print(
                Panel(
                    f"[bold yellow]⚠ Authentication Required[/bold yellow]\n\n"
                    f"The API requires authentication.\n"
                    f"Add MESSAGES_API_KEY to your .env file.",
                    border_style="yellow",
                    title="[bold]Status: Auth Needed[/bold]"
                )
            )
            return False
        else:
            console.print(
                Panel(
                    f"[bold red]✗ API Error[/bold red]\n\n"
                    f"Status Code: {response.status_code}\n"
                    f"Response: {response.text[:200]}",
                    border_style="red",
                    title="[bold]Status: Error[/bold]"
                )
            )
            return False
            
    except requests.exceptions.Timeout:
        console.print(
            Panel(
                f"[bold red]✗ Timeout[/bold red]\n\n"
                f"The API did not respond within 10 seconds.",
                border_style="red",
                title="[bold]Status: Timeout[/bold]"
            )
        )
        return False
    except requests.exceptions.ConnectionError:
        console.print(
            Panel(
                f"[bold red]✗ Connection Error[/bold red]\n\n"
                f"Could not connect to the API.\n"
                f"Check your internet connection.",
                border_style="red",
                title="[bold]Status: Connection Error[/bold]"
            )
        )
        return False
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]✗ Error[/bold red]\n\n"
                f"{str(e)}",
                border_style="red",
                title="[bold]Status: Error[/bold]"
            )
        )
        return False


if __name__ == "__main__":
    success = check_api()
    console.print()
    exit(0 if success else 1)

