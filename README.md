# Adrian Code Agent

A lightweight local coding agent built in Python. It runs in the terminal, lets the model read files, write files, and execute shell commands, and is designed to help with code generation and small software tasks.

## Features

- Terminal-based interactive workflow
- Tool calling for reading and writing files
- Shell command execution
- Model API call tracking for debugging
- Cross-platform path handling for Windows and Unix-like systems

## Project structure

- `main.py` — CLI entry point
- `agent/` — model setup, tools, hooks
- `ui/` — terminal rendering and commands
- `tests/` — regression tests

## Requirements

- Python 3.10+
- An API key for the configured model provider

## Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If you do not yet have a `requirements.txt`, install the project dependencies manually, for example:

   ```bash
   pip install pydantic-ai rich prompt_toolkit
   ```

3. Set your API key for the model provider:

   PowerShell:

   ```powershell
   $env:API_KEY="your_api_key_here"
   ```

   Bash/zsh:

   ```bash
   export API_KEY="your_api_key_here"
   ```

4. Run the app:

   ```bash
   python main.py
   ```

## Common commands

Inside the app, you can use commands such as:

- `/help` — list available commands
- `/status` — show session status
- `/new` — start a fresh session
- `/api-detail` — show recent model API calls
- `/exit` — quit the program

## Notes

This project is intended as a local tool/experiment and should be used with care when allowing the model to run shell commands or modify files.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
