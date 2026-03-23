# Shared ignore rules for Atenea Server

IGNORED_DIRS = {
    ".git", "build", "node_modules", ".gradle", ".venv", "venv", 
    ".idea", "bin", "obj", "out", "metadata", ".metadata", ".next", "dist", 
    "target", "__pycache__", ".vscode", ".pytest_cache", ".mypy_cache"
}

BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", 
    ".exe", ".dll", ".so", ".bin", ".jar", ".class", ".aar", ".xcf",
    ".svg", ".ttf", ".otf", ".woff", ".woff2", ".7z", ".tar", ".gz",
    ".dmg", ".iso", ".sqlite"
}

IGNORED_FILES = {
    "gradlew", "gradlew.bat", 
    ".gitignore", "gradle.properties", "settings.gradle", "package-lock.json",
    "yarn.lock", "pnpm-lock.yaml", ".DS_Store"
}

