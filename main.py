import argparse
import os
import git
import json
import time
import requests
from typing import List, Dict

SYSTEM_PROMPT = """
You are an expert software developer helping to write clear, concise, and informative commit messages.
Analyze the git diff changes and generate a commit message that follows conventional commit format.

Guidelines:
- Use conventional commit format: <type>(<scope>): <description>
- Common types: feat, fix, docs, style, refactor, test, chore
- Keep the description under 50 characters
- Provide a detailed body explaining WHAT changed and WHY (if needed)
- Focus on the actual changes, not the process
- Be specific about what was added, removed, or modified

Example format:
feat(auth): add user login functionality

- Implement JWT token generation
- Add login endpoint at /api/auth/login
- Include input validation for credentials
"""

MODEL_NAME = "llama3"
API_ENDPOINT = "http://localhost:11434/v1/chat/completions"
API_KEY = "ollama"
MAX_RETRIES = 3
BASE_DELAY = 1


def is_git_repository(path: str) -> bool:
    try:
        _ = git.Repo(path).git_dir
        return True
    except git.exc.InvalidGitRepositoryError:
        return False


def get_git_diffs(repo_path: str) -> str:
    """
    Get the git diffs for the repository and preprocess them for the LLM.
    """
    repo = git.Repo(repo_path)
    diffs = []

    # Get staged changes
    for item in repo.index.diff(None):
        try:
            diff_text = repo.git.diff(item.a_path)

            # Preprocess the diff to make it more LLM-friendly
            processed_diff = []
            for line in diff_text.split("\n"):
                # Keep only relevant diff lines (changes, file headers, and context)
                if (line.startswith('+') or line.startswith('-') or
                        line.startswith('@@') or line.startswith('diff --git') or
                        line.startswith('---') or line.startswith('+++') or
                        line.startswith('index ')):
                    processed_diff.append(line)

            if processed_diff:
                file_diff = f"File: {item.a_path}\n" + "\n".join(processed_diff)
                diffs.append(file_diff)

        except Exception as e:
            print(f"Warning: Could not get diff for {item.a_path}: {e}")
            continue

    return "\n\n".join(diffs)


def generate_commit_message(diffs: str) -> str | None:
    """
    Generate a commit message from the git diffs using the LLM.
    """
    user_prompt = f"""
Please analyze the following git changes and generate an appropriate commit message:

{diffs}

Remember to follow the conventional commit format and focus on the actual changes made.
"""

    response = call_llm_api(user_prompt)
    return response


def call_llm_api(prompt: str) -> str | None:
    """
    Call the LLM API with retry logic and proper error handling.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}" if API_KEY else ""
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,  # Lower temperature for more consistent results
        "max_tokens": 500
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_ENDPOINT, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    print(f"Unexpected response format: {data}")
                    return None
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Request error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")

        # Exponential backoff
        if attempt < MAX_RETRIES - 1:
            delay = BASE_DELAY * (2 ** attempt)
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)

    print("Failed to get response from LLM after all retries.")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate commit message suggestions using an LLM."
    )
    parser.add_argument("repo_path", help="Path to the git repository")
    args = parser.parse_args()

    if not is_git_repository(args.repo_path):
        print("Error: The specified path is not a valid git repository.")
        return

    diffs = get_git_diffs(args.repo_path)
    if not diffs:
        print("No changes detected in the repository.")
        return

    try:
        commit_message = generate_commit_message(diffs)
        if commit_message is None:
            print("No commit message generated.")
            return

        print("Suggested commit message:")
        print("=" * 50)
        print(commit_message)
        print("=" * 50)

    except Exception as e:
        print(f"An error occurred while generating the commit message: {str(e)}")
        print("Please try again later or write the commit message manually.")


if __name__ == "__main__":
    main()