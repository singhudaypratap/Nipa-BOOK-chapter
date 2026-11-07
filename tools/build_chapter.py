#!/usr/bin/env python3
"""Build a book chapter from a keyed Markdown source.

Source files use citation keys of the form [@key] or [@key1; @key2].
This script:
  1. numbers references in order of first appearance (IEEE convention),
  2. rewrites keys to IEEE brackets, e.g. [3], [2, 5], [4]-[6] ranges,
  3. appends the numbered reference list with DOIs,
  4. fails loudly if a key is unknown or a reference is never cited,
  5. reports word counts (body text, excluding references and code),
  6. writes <name>.md and, if pandoc is present, <name>.docx.

Usage: python3 tools/build_chapter.py <chapter_dir>
The chapter dir must contain parts/*.md (concatenated in name order) and references.json.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

CITE = re.compile(r"\[(@[A-Za-z0-9_]+(?:\s*;\s*@[A-Za-z0-9_]+)*)\]")


def fmt_numbers(nums):
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            parts.append(f"{nums[i]}–{nums[j]}")
        else:
            parts.extend(str(n) for n in nums[i:j + 1])
        i = j + 1
    return "[" + ", ".join(parts) + "]"


def main(chapter_dir):
    d = Path(chapter_dir)
    parts = sorted((d / "parts").glob("*.md"))
    src = "\n\n".join(p.read_text(encoding="utf-8").strip() for p in parts) + "\n"
    # Word's equation editor has no \tag; render equation numbers explicitly.
    src = re.sub(r"\\tag\{([^}]*)\}", r"\\qquad (\1)", src)
    refs = json.loads((d / "references.json").read_text(encoding="utf-8"))
    out_name = refs.pop("_output_name")

    order = []
    for m in CITE.finditer(src):
        for key in re.findall(r"@([A-Za-z0-9_]+)", m.group(1)):
            if key not in refs:
                sys.exit(f"ERROR: unknown citation key '{key}'")
            if key not in order:
                order.append(key)
    unused = [k for k in refs if k not in order]
    if unused:
        sys.exit(f"ERROR: references never cited: {unused}")
    num = {k: i + 1 for i, k in enumerate(order)}

    body = CITE.sub(
        lambda m: fmt_numbers(num[k] for k in re.findall(r"@([A-Za-z0-9_]+)", m.group(1))),
        src,
    )

    ref_lines = ["", "## References", ""]
    for k in order:
        r = refs[k]
        r["citation"] = r["citation"].replace("<i>", "*").replace("</i>", "*")
        ref_lines.append(f"[{num[k]}] {r['citation']} doi: [{r['doi']}](https://doi.org/{r['doi']})")
        ref_lines.append("")
    full = body.rstrip() + "\n" + "\n".join(ref_lines)

    out_md = d / f"{out_name}.md"
    out_md.write_text(full, encoding="utf-8")

    # Word count of the narrative: drop code blocks, math, tables' pipes, image lines.
    text = re.sub(r"```.*?```", " ", body, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\$[^$\n]+\$", " X ", text)
    text = text.replace("|", " ")
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-–/.]*", text)]
    print(f"{out_md}: {len(words)} words (body, excl. references), {len(order)} references cited")

    if shutil.which("pandoc"):
        out_docx = d / f"{out_name}.docx"
        subprocess.run(
            ["pandoc", out_md.name, "-o", out_docx.name, "--from", "markdown+tex_math_dollars+pipe_tables",
             "--resource-path", "."],
            cwd=d, check=True,
        )
        print(f"{out_docx}: written")


if __name__ == "__main__":
    main(sys.argv[1])
