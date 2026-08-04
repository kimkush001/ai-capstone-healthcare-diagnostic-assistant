"""
Scans every file in modules/ for typing hints (Dict, List, Optional, etc.)
that are used but never imported from Python's typing module.
"""
import ast
import glob

TYPING_NAMES = {'Dict', 'List', 'Optional', 'Tuple', 'Set', 'Union', 'Any', 'Callable'}

def check_file(path):
    with open(path, encoding='utf-8') as f:
        source = f.read()

    tree = ast.parse(source, filename=path)

    used_names = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }

    imported_typing_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'typing':
            imported_typing_names |= {alias.name for alias in node.names}

    used_typing = used_names & TYPING_NAMES
    missing = used_typing - imported_typing_names

    return missing

def main():
    files = sorted(glob.glob('modules/*.py'))
    if not files:
        print("No files found in modules/ — are you running this from the project root?")
        return

    any_missing = False
    for path in files:
        missing = check_file(path)
        if missing:
            any_missing = True
            print(f"❌ {path}")
            print(f"   Missing typing imports: {sorted(missing)}")
        else:
            print(f"✅ {path}")

    print()
    if any_missing:
        print("Some files are missing typing imports — fix them before merging.")
    else:
        print("All files import every typing name they use. 🎉")

if __name__ == "__main__":
    main()