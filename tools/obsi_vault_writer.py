"""Vault writer: Creates structured markdown notes optimized for Obsidian Graph View."""

import json
import os
import re
import unicodedata

DATA_DIR = os.path.abspath(os.path.expanduser(os.environ.get("SELENE_DATA_DIR", "~/.selene-agent")))
VAULTS_DIR = os.path.join(DATA_DIR, "vaults")

def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _safe_link(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").replace("]]", "] ]").strip()


def _as_list(value: object) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]

def create_structured_note(
    title: str = "Untitled Note", 
    content: str = "", 
    incoming_links: list[str] | None = None,
    outgoing_links: list[str] | None = None,
    tags: list[str] | None = None,
) -> str:
    """
    Creates a structured Obsidian note optimized for Graph View.
    
    This function generates a markdown file formatted specifically for knowledge
    graph tools like Obsidian. It includes YAML frontmatter for tags, safe filename
    generation to prevent overwrites of existing notes, and automatic appending or
    prepending of cross-reference (wiki) links.
    
    Args:
        title (str): The primary header and base filename for the note.
        content (str): The main body text of the note.
        incoming_links (list[str]): Optional wiki-links to place at the top.
        outgoing_links (list[str]): Optional wiki-links to place in a 'Related Concepts' section.
        tags (list[str]): Optional string tags to place in the YAML frontmatter.
        
    Returns:
        str: A JSON-encoded string indicating success with the created filepath
             or an error message on failure.
    """
    if not title:
        title = "Untitled Note"
    else:
        title = str(title)
    display_title = re.sub(r"[\r\n\x00-\x1f]+", " ", title).strip() or "Untitled Note"
        
    if not content:
        content = ""
    else:
        content = str(content)
        
    os.makedirs(VAULTS_DIR, exist_ok=True)
    
    # 2. Graph View Optimization: Sanitize file names (replace invalid characters with dashes)
    safe_title = unicodedata.normalize("NFKC", display_title)
    safe_title = re.sub(r'[\\/*?:"<>|\x00-\x1f]', '-', safe_title)
    safe_title = re.sub(r'\s+', ' ', safe_title).strip()
    safe_title = safe_title.strip(". ")[:180] or "Untitled Note"
    if safe_title.casefold() in {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}:
        safe_title = f"_{safe_title}"
    
    # 4. Prevent Destructive Overwrites: versioned filename
    base_filename = f"{safe_title}.md"
    filepath = os.path.join(VAULTS_DIR, base_filename)
        
    note_lines = []
    
    # Format tags into standard YAML frontmatter
    tags = _as_list(tags)
    incoming_links = _as_list(incoming_links)
    outgoing_links = _as_list(outgoing_links)

    if tags:
        note_lines.append("---")
        note_lines.append("tags:")
        clean_tags = (str(tag).strip().strip('#') for tag in tags)
        for tag in dict.fromkeys(tag for tag in clean_tags if tag):
            note_lines.append(f"  - {json.dumps(tag, ensure_ascii=False)}")
        note_lines.append("---")
        note_lines.append("")
        
    note_lines.append(f"# {display_title}")
    note_lines.append("")
    
    # 3. Internal Linking: Append or prepend cross-reference links
    if incoming_links:
        links = [_safe_link(link) for link in incoming_links]
        note_lines.append("**Incoming Links:** " + ", ".join(f"[[{link}]]" for link in links if link))
        note_lines.append("")
        
    note_lines.append(content)
    note_lines.append("")
    
    # If outgoing_links are provided, format into Related Concepts section at the bottom
    if outgoing_links:
        note_lines.append("## Related Concepts")
        for link in outgoing_links:
            clean_link = _safe_link(link)
            if clean_link:
                note_lines.append(f"- [[{clean_link}]]")
        note_lines.append("")
        
    final_content = "\n".join(note_lines)
    
    try:
        counter = 2
        while True:
            try:
                with open(filepath, "x", encoding="utf-8") as f:
                    f.write(final_content)
                break
            except FileExistsError:
                filepath = os.path.join(VAULTS_DIR, f"{safe_title} v{counter}.md")
                counter += 1
    except Exception as e:
        return _json({"error": f"Failed to write note: {str(e)}"})
        
    return _json({
        "status": "success",
        "filepath": filepath,
        "title": title,
        "message": f"Structured note '{os.path.basename(filepath)}' created successfully."
    })
