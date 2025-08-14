# ai-agent-challenge
Coding agent challenge which write custom parsers for Bank statement PDF.

**An AI agent that writes, tests, and self-fixes custom parsers for bank statement PDFs — with a robust fallback when things get messy.**

---
[Watch the video on Google Drive](https://drive.google.com/file/d/1GSs94P0cYnqE1kSrK1uC3ayJiXjGiV7E/view?usp=sharing)

## Overview
This project implements an **AI agent** that automatically generates **custom Python parsers** for **bank statement PDFs**. The workflow is orchestrated with **LangGraph**, while **Groq LLMs** generate and iteratively refine parser code. If a generated parser fails tests after multiple repair attempts, the agent **falls back to a pre-verified robust parser**.

---

## Features
- **Automated Parser Generation**: Creates tailor‑made Python parsers for bank statement PDFs.
- **Self‑Correction**: Runs tests and applies fixes up to **3** times.
- **Fallback Mechanism**: Uses a **pre‑verified robust parser** on repeated failure.
- **Multi‑Format Support**: Handles diverse layouts and formats across banks.
- **Test Automation**: Includes **pytest** suites to verify correctness.

---

## Project Structure
    ```text
    ai-agent-challenge/
    ├── agent.py                 # Main agent implementation
    ├── custom_parsers/          # Generated parsers directory
    │   └── icici_parser.py      # Example generated parser
    ├── data/
    │   └── icici/
    │       ├── icici_sample.pdf # Sample ICICI bank statement
    │       └── icici_sample.csv # Expected CSV output
    ├── tests/
    │   └── test_parsers.py      # Parser tests
    ├── debug_parser.py          # Debug script to compare outputs
    ├── requirements.txt         # Dependencies
    ├── .env                     # Environment variables (not committed)
    └── README.md                # This file

## Installation
1.**Clone the repository And Run Agent**
  ```bash
  git clone https://github.com/Anshad-Aziz/ai-agent-challenge
  cd ai-agent-challenge
  python agent.py --target {bank_name} --model {model_name}

