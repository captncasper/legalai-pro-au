import numpy as np
#!/usr/bin/env python3
"""Command line interface for Australian Legal AI"""
import click
import requests
import json
import os


def get_api_url():
    """Get API URL based on environment"""
    if os.environ.get('CODESPACES'):
        codespace_name = os.environ.get('CODESPACE_NAME', '')
        return f"https://{codespace_name}-8000.preview.app.github.dev"
    return "http://localhost:8000"


@click.group()
def cli():
    """Australian Legal AI CLI"""
    pass


@cli.command()
@click.option('--query', '-q', required=True, help='Legal question to search')
@click.option('--api-key', '-k', default='demo_key', help='API key')
def search(query, api_key):
    """Search legal documents"""
    base_url = get_api_url()
    response = requests.post(
        f"{base_url}/search",
        json={"query": query},
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    if response.status_code == 200:
        results = response.json()
        click.echo(f"\nResults for: {results['query']}\n")
        for i, r in enumerate(results['results'], 1):
            click.echo(f"{i}. Score: {r['relevance_score']:.3f}")
            click.echo(f"   {r['document_excerpt']}\n")
    else:
        click.echo(f"Error: {response.status_code}")
        click.echo(f"URL: {base_url}")


if __name__ == "__main__":
    cli()
