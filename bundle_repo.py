#!/usr/bin/env python3
import os
import sys
import base64

# Configuration
# 45 MB limit (in bytes)
MAX_FILE_SIZE = 45 * 1024 * 1024 

# Extensions we'll treat as text and dump directly
TEXT_EXTS = {
    ".txt", ".py", ".md", ".rst", ".csv",
    ".json", ".yaml", ".yml", ".ini", ".cfg",
    ".toml", ".xml", ".html", ".htm", ".css",
    ".js", ".ts", ".ipynb"
}

# Directories to skip
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".idea", ".DS_Store"}

def is_text_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in TEXT_EXTS

def main():
    if len(sys.argv) < 2:
        print("Usage: python bundle_repo.py <root_dir> [output_base_name]")
        sys.exit(1)

    root_dir = sys.argv[1]
    # Default base name provided by user or default string
    base_name_input = sys.argv[2] if len(sys.argv) > 2 else "repo_bundle.txt"
    
    # Prepare to track file parts
    name_root, name_ext = os.path.splitext(base_name_input)
    part_num = 1
    
    # Open first file
    current_out_name = f"{name_root}_part{part_num}{name_ext}"
    out_file = open(current_out_name, "w", encoding="utf-8")
    
    # Write initial header and track size
    header = f"# Repo bundle for: {os.path.abspath(root_dir)}\n# Part {part_num}\n\n"
    out_file.write(header)
    current_size = len(header.encode('utf-8'))

    print(f"Bundling... (Output limit: {MAX_FILE_SIZE / (1024*1024):.2f} MB per file)")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Prune dirs
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in sorted(filenames):
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, root_dir)
            ext = os.path.splitext(fname)[1].lower() or "<no-ext>"

            # Buffer the content string for this specific file
            entry_buffer = []
            entry_buffer.append("\n" + "=" * 80 + "\n")
            entry_buffer.append(f"FILE: {rel_path}\n")
            entry_buffer.append(f"EXT:  {ext}\n")
            entry_buffer.append("=" * 80 + "\n")

            try:
                if is_text_file(full_path):
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        entry_buffer.append(f.read())
                else:
                    with open(full_path, "rb") as f:
                        data = f.read()
                    b64 = base64.b64encode(data).decode("ascii")
                    entry_buffer.append("\n[BINARY FILE – base64 encoded below]\n\n")
                    entry_buffer.append(b64)
                    entry_buffer.append("\n")
            except Exception as e:
                entry_buffer.append(f"\n[ERROR reading file: {e}]\n")

            # Combine buffer to check size
            full_entry_str = "".join(entry_buffer)
            entry_size_bytes = len(full_entry_str.encode('utf-8'))

            # Check if writing this entry would exceed the limit
            # We only rotate if current_size > 0 to avoid infinite loops on huge single files
            if (current_size + entry_size_bytes > MAX_FILE_SIZE) and (current_size > 0):
                out_file.close()
                print(f"Finished {current_out_name} ({current_size / (1024*1024):.2f} MB)")
                
                # Start new part
                part_num += 1
                current_out_name = f"{name_root}_part{part_num}{name_ext}"
                out_file = open(current_out_name, "w", encoding="utf-8")
                
                # Write new header
                new_header = f"# Repo bundle for: {os.path.abspath(root_dir)}\n# Part {part_num} (Continued)\n\n"
                out_file.write(new_header)
                current_size = len(new_header.encode('utf-8'))

            # Write the entry
            out_file.write(full_entry_str)
            current_size += entry_size_bytes

    out_file.close()
    print(f"Done. Final part: {current_out_name} ({current_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()