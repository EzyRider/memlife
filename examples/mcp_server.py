"""Standalone MCP server example.

Run: python examples/mcp_server.py --db ./mem.db --embedder ollama --model mxbai-embed-large:latest

Then add to your Claude Desktop config (or any MCP client):
{
  "mcpServers": {
    "memlife": {
      "command": "python",
      "args": ["/path/to/examples/mcp_server.py", "--db", "./mem.db"],
      "env": {}
    }
  }
}
"""

from memlife.mcp_server import main

if __name__ == "__main__":
    main()