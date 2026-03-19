import os
import subprocess
import sys
import platform

def run_command(command, shell=True):
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=shell)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        return False
    return True

def setup():
    is_windows = platform.system() == "Windows"
    python_cmd = "python" if is_windows else "python3"
    venv_dir = ".venv"
    
    print(f"--- Setting up Atenea Server ({platform.system()}) ---")

    # 1. Create Venv
    if not os.path.exists(venv_dir):
        if not run_command(f"{python_cmd} -m venv {venv_dir}"):
            return

    # 2. Install dependencies
    pip_path = os.path.join(venv_dir, "Scripts", "pip") if is_windows else os.path.join(venv_dir, "bin", "pip")
    if not run_command(f"{pip_path} install -e ."):
        return

    # 3. Docker Compose
    print("--- Starting Qdrant via Docker Compose ---")
    if not run_command("docker compose up -d"):
        print("Warning: Docker Compose failed. Make sure Docker is running.")

    # 4. Ollama
    print("--- Pulling Nomic Embed Text via Ollama ---")
    if not run_command("ollama pull nomic-embed-text"):
        print("Warning: Ollama pull failed. Make sure Ollama is installed and running.")

    print("\n--- Setup Complete! ---")
    print(f"To run the server, use: {os.path.join(venv_dir, 'Scripts', 'atenea-server') if is_windows else os.path.join(venv_dir, 'bin', 'atenea-server')}")

if __name__ == "__main__":
    setup()
