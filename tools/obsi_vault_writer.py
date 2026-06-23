"""Vault writer: Creates structured markdown notes optimized for Obsidian Graph View."""

import json
import os
import re

VAULTS_DIR = os.path.expanduser("~/.selene-agent/vaults")

def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)

def create_structured_note(
    title: str = "Untitled Note", 
    content: str = "", 
    incoming_links: list[str] = None, 
    outgoing_links: list[str] = None, 
    tags: list[str] = None
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
        
    if not content:
        content = ""
    else:
        content = str(content)
        
    os.makedirs(VAULTS_DIR, exist_ok=True)
    
    # 2. Graph View Optimization: Sanitize file names (replace invalid characters with dashes)
    safe_title = re.sub(r'[\\/*?:"<>|]', '-', title)
    safe_title = re.sub(r'\s+', ' ', safe_title).strip()
    
    # 4. Prevent Destructive Overwrites: versioned filename
    base_filename = f"{safe_title}.md"
    filepath = os.path.join(VAULTS_DIR, base_filename)
    counter = 2
    while os.path.exists(filepath):
        filepath = os.path.join(VAULTS_DIR, f"{safe_title} v{counter}.md")
        counter += 1
        
    note_lines = []
    
    # Format tags into standard YAML frontmatter
    if tags:
        note_lines.append("---")
        note_lines.append("tags:")
        for tag in tags:
            clean_tag = tag.strip().strip('#')
            note_lines.append(f"  - {clean_tag}")
        note_lines.append("---")
        note_lines.append("")
        
    note_lines.append(f"# {title}")
    note_lines.append("")
    
    # 3. Internal Linking: Append or prepend cross-reference links
    if incoming_links:
        note_lines.append("**Incoming Links:** " + ", ".join([f"[[{link}]]" for link in incoming_links]))
        note_lines.append("")
        
    note_lines.append(content)
    note_lines.append("")
    
    # If outgoing_links are provided, format into Related Concepts section at the bottom
    if outgoing_links:
        note_lines.append("## Related Concepts")
        for link in outgoing_links:
            note_lines.append(f"- [[{link}]]")
        note_lines.append("")
        
    final_content = "\n".join(note_lines)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(final_content)
    except Exception as e:
        return _json({"error": f"Failed to write note: {str(e)}"})
        
    return _json({
        "status": "success",
        "filepath": filepath,
        "title": title,
        "message": f"Structured note '{os.path.basename(filepath)}' created successfully."
    })
