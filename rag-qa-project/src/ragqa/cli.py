"""Usage:
    python -m ragqa.cli build ./data/raw
    python -m ragqa.cli ask "What did the reviews say about late deliveries?"
"""
from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown

from ragqa.pipeline import ask, build_index

console = Console()


def main() -> None:
    if len(sys.argv) < 2:
        console.print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "build":
        data_dir = sys.argv[2] if len(sys.argv) > 2 else "./data/raw"
        build_index(data_dir)

    elif command == "ask":
        if len(sys.argv) < 3:
            console.print("[red]Provide a question in quotes.[/red]")
            sys.exit(1)
        question = sys.argv[2]
        result = ask(question)
        console.print(Markdown(f"**Q:** {result['question']}\n\n**A:** {result['answer']}"))
        console.print("\n[bold]Sources:[/bold]")
        for s in result["sources"]:
            console.print(f"  [{s['index']}] {s['source']}  (similarity={s['similarity']})")

    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
