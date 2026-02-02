import re
from pathlib import Path
import pymupdf4llm
from typing import List, Dict, Optional
import unicodedata

# ────────────────────────────────────────────────
# Part 1
# ────────────────────────────────────────────────

def pdf_to_markdown(
    pdf_path: str | Path,
    output_md: str | Path = "output.md",
    image_folder: str | Path = "images",
    image_format: str = "png",
    # dpi: int = 150,
    remove_header: bool = True,
    remove_footer: bool = True,
    page_chunks: bool = True,
    add_page_comments: bool = True,
) -> int:
    """
    Convert a PDF file to Markdown with extracted images.

    Args:
        pdf_path: Path to the input PDF file (str or Path)
        output_md: Path where the Markdown file will be saved
        image_folder: Folder where extracted images will be saved
        image_format: Image format ("png", "jpeg", "webp")
        dpi: Resolution for rendered images (lower = smaller files)
        remove_header: Try to exclude detected page headers
        remove_footer: Try to exclude detected page footers
        page_chunks: Process page-by-page (recommended for large documents)
        add_page_comments: Insert <!-- Page N --> markers in the output

    Returns:
        Number of pages processed (or -1 on failure)

    Raises:
        FileNotFoundError: If PDF does not exist
        RuntimeError: On conversion failure
    """
    pdf_path = Path(pdf_path)
    output_md = Path(output_md)
    image_folder = Path(image_folder)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Ensure image output directory exists
    image_folder.mkdir(parents=True, exist_ok=True)

    try:
        md_chunks: List[Dict] = pymupdf4llm.to_markdown(
            str(pdf_path),                      # must be string for pymupdf
            page_chunks=page_chunks,
            write_images=True,
            image_path=str(image_folder),
            image_format=image_format,
            # dpi=dpi,
            page_separators=True,
            header=not remove_header,           # API is inverted: True = keep
            footer=not remove_footer,
        )

        if not md_chunks:
            print("Warning: No content extracted from PDF.")
            return 0

        # Build final markdown content
        if add_page_comments:
            full_md = ""
            for i, chunk in enumerate(md_chunks, 1):
                full_md += f"<!-- Page {i} -->\n{chunk['text'].rstrip()}\n\n"
        else:
            full_md = "\n\n".join(chunk["text"].rstrip() for chunk in md_chunks)

        return full_md

    except Exception as e:
        print(f"Error converting PDF → {pdf_path.name}")
        print(f"→ {type(e).__name__}: {e}")
        return 





# ────────────────────────────────────────────────
# Part 2
# ────────────────────────────────────────────────

# --- CONFIGURATION ---
# You can easily add/remove from this list using natural symbols
ALLOWED_PATTERNS = [
    "[0-9].",
    # "[a-z])",
    # "[A-Z])",
    # "[0-9])"
]

def group_lines_into_blocks(text: str) -> list[str]:
    blocks = []
    current = []

    def is_table_separator(line: str) -> bool:
        # Check for standard markdown table separator: |---|---|
        # It must have pipes and hyphens.
        return '|' in line and set(line.replace('|', '').strip()) <= {'-', ':', ' '}

    def process_buffer(lines):
        if not lines:
            return

        # Check if any line in the current block looks like a table separator
        is_table = any(is_table_separator(line) for line in lines)

        if is_table:
            # If it's a table, keep lines separate (join with newline)
            blocks.append("\n".join(lines))
        else:
            # If it's text, merge them (join with space)
            blocks.append(" ".join(lines))

    for line in text.splitlines():
        if line.strip():  # if there's any non-whitespace content
            current.append(line.strip())
        elif current:
            process_buffer(current)
            current = []

    # Handle the last block
    if current:
        process_buffer(current)

    return blocks

def parse_user_patterns(patterns):
    """
    Converts user-friendly strings like '[0-9].' into regex like '[0-9]+\.'
    """
    regex_list = []
    for p in patterns:
        # 1. Escape the whole string first (handles . and ) automatically)
        # 2. If it contains [0-9], add a '+' so it matches '10.' as well as '1.'
        escaped = re.escape(p).replace(r'\[0\-9\]', r'[0-9]+')
        # Ensure bracket groups [a-z] are restored from escaping
        escaped = escaped.replace(r'\[a\-z\]', r'[a-z]').replace(r'\[A\-Z\]', r'[A-Z]')
        regex_list.append(escaped)

    # Combine into: ^\s*(pattern1|pattern2)\s+
    return rf"^\s*({'|'.join(regex_list)})\s+.*"

def bold_markdown_lists(content):
    # Parse the user patterns into a master regex
    master_pattern = parse_user_patterns(ALLOWED_PATTERNS)

    try:
        # with open(file_path, 'r', encoding='utf-8') as f:
        #     lines = f.readlines()
        lines = group_lines_into_blocks(content)

        new_lines = []
        for line in lines:
            stripped = line.strip()

            # Match against the allowed patterns
            is_match = re.match(master_pattern, line)
            # Prevent double bolding
            is_already_bold = stripped.startswith("**") and stripped.endswith("**")

            if is_match and not is_already_bold:
                content = line.rstrip('\n')
                new_lines.append(f"\n**{content}**\n")
            else:
                new_lines.append(f"\n{line}")

        # with open(out_path, 'w', encoding='utf-8') as f:
        #     f.writelines(new_lines)

        # print(f"Successfully processed '{file_path}'")
        print(f"Applied Patterns: {ALLOWED_PATTERNS}")

        return "\n".join(new_lines)

    except FileNotFoundError:
        print(f"Error")


def merge_multiline_bold_tags(markdown_text: str) -> str:
    """
    Fixes bold text that has been split across multiple lines.
    Example:
    **Line 1**
    **Line 2**
    -> **Line 1 Line 2**
    """
    # Pattern 1: Matches two consecutive lines both wrapped in asterisks
    # Example: **Text** \n **Text**
    re_double_wrapped = r'(?m)^([ \t]*\*\*.*?\*\*)[ \t]*\n[ \t]*(\*\*.*?\*\*)[ \t]*$'

    # Pattern 2: Matches a line starting with ** and a following line ending with **
    # Example: **Text \n **More Text**
    re_partial_wrapped = r'(?m)^([ \t]*\*\*.*?)[ \t]*\n[ \t]*(\*\*.*?\*\*)[ \t]*$'

    def replace_double_wrapped(match):
        part1 = match.group(1).strip()
        part2 = match.group(2).strip()
        # Remove trailing ** from part1 and leading ** from part2
        return f"{part1[:-2].strip()} {part2[2:].strip()}"

    def replace_partial_wrapped(match):
        part1 = match.group(1).strip()
        part2 = match.group(2).strip()
        # Keep part1 as is, remove leading ** from part2
        return f"{part1} {part2[2:].strip()}"

    # Loop to handle 3+ line splits (iteratively merges lines)
    while True:
        previous_text = markdown_text
        markdown_text = re.sub(re_double_wrapped, replace_double_wrapped, markdown_text)
        markdown_text = re.sub(re_partial_wrapped, replace_partial_wrapped, markdown_text)

        if markdown_text == previous_text:
            break

    return markdown_text


def merge_inline_fragmented_bold(markdown_text: str) -> str:
    """
    Fixes consecutive bold tags on the same line by merging them into a single bold block.
    Example:
    **Điều 7. Thời hiệu xử phạt vi phạm hành chính và thời hạn được coi** **là chưa bị xử phạt vi phạm hành chính**
    ->
    **Điều 7. Thời hiệu xử phạt vi phạm hành chính và thời hạn được coi là chưa bị xử phạt vi phạm hành chính**

    Handles multiple consecutive bolds (e.g., **a** **b** **c** → **a b c**).
    Normalizes spacing and only merges within the same line.
    """
    # Pattern matches **content** followed by optional horizontal whitespace then another **content**
    pattern = r'\*\*([^\*]+)\*\*[ \t]*\*\*([^\*]+)\*\*'

    def replace(match):
        part1 = match.group(1).strip()
        part2 = match.group(2).strip()
        return f"**{part1} {part2}**"

    while True:
        previous_text = markdown_text
        markdown_text = re.sub(pattern, replace, markdown_text)
        if markdown_text == previous_text:
            break

    return markdown_text

def repair_markdown_file(content: str):
    """
    Reads a markdown file, repairs broken bold formatting, and saves the result.
    """
    # input_file = Path(input_path)
    # if not input_file.exists():
    #     print(f"Error: file '{input_path}' not found.")
    #     return

    # 1. Read
    # content = input_file.read_text(encoding="utf-8")

    # 1. Add sub-sections if exits
    content = bold_markdown_lists(content)

    # 2. Process: Fix vertical splits then horizontal splits
    content = merge_multiline_bold_tags(content)
    content = merge_inline_fragmented_bold(content)

    # 3. Write
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     f.write(content)

    print(f"Successfully repaired bolding!")
    # print(f"Output saved to: {output_path}")
    return content

# ────────────────────────────────────────────────
# Part 3
# ────────────────────────────────────────────────
def slugify(text: str) -> str:
    """Tạo anchor link chuẩn (không dấu, lowercase)"""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'\s+', '-', text.strip())
    return text

def add_toc_to_markdown(
    text: str,
    add_toc: bool = True,
    # output_path: str = None,
    heading_rules: list[dict] = None,
):  
    lines = text.splitlines()

    # Chuẩn hóa rules map
    # Ví dụ: {'chương': 2, 'điều': 3, 'digit': 4}
    rules_map = {}
    if heading_rules:
        for r in heading_rules:
            rules_map[r["prefix"].lower()] = r["level"]

    new_lines = []
    headings = []

    # Regex 1: Chỉ bắt những dòng ĐƯỢC BÔI ĐẬM TOÀN BỘ
    # ^\s*    : Đầu dòng + khoảng trắng tùy ý
    # \*\*    : Dấu mở bôi đậm
    # (.+?)   : Nội dung bên trong (Group 1)
    # \*\*    : Dấu đóng bôi đậm
    # \s*$    : Khoảng trắng + Cuối dòng
    BOLD_LINE_REGEX = r'^\s*\*\*(.+?)\*\*\s*$'

    # Regex 2: Kiểm tra nội dung bên trong bắt đầu bằng gì
    # (Chương|...|\d+) : Bắt đầu bằng từ khóa hoặc số
    PREFIX_CHECK_REGEX = r'^(Chương|Mục|Điều|Phụ lục|(?:\d+))'

    for line in lines:
        stripped = line.strip()
        new_line = line
        level = 0
        title = ""

        # --- BƯỚC 1: Kiểm tra xem dòng có phải là **...** không ---
        bold_match = re.match(BOLD_LINE_REGEX, stripped)

        is_transformed = False

        if bold_match:
            # Lấy nội dung bên trong cặp ** (ví dụ: "Điều 5. Cái gì đó")
            inner_content = bold_match.group(1).strip()

            # --- BƯỚC 2: Kiểm tra nội dung đó có bắt đầu bằng Keyword/Số không ---
            prefix_match = re.match(PREFIX_CHECK_REGEX, inner_content, re.IGNORECASE)

            if prefix_match:
                prefix_found = prefix_match.group(1) # VD: Điều, 10, Chương

                # Xác định key để tra cứu rule
                key = "digit" if prefix_found.isdigit() else prefix_found.lower()

                if key in rules_map:
                    level = rules_map[key]
                    title = inner_content # Giữ nguyên nội dung (Điều 5. ABC)

                    # TẠO DÒNG MỚI: Bỏ dấu **, thêm dấu #
                    new_line = f"{'#' * level} {title}"
                    is_transformed = True

        # Nếu không phải dòng custom heading, check heading Markdown thường (#)
        if not is_transformed:
            m_std = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            if m_std:
                level = len(m_std.group(1))
                title = m_std.group(2).strip()
                new_line = line.rstrip()

        new_lines.append(new_line)

        # Tạo mục lục nếu là heading
        if level > 0 and title:
            anchor = slugify(title)
            headings.append((level, title, anchor))

    # --- Phần ghi file và tạo TOC (như cũ) ---
    toc_lines = ["## MỤC LỤC", ""]
    for lvl, txt, anc in headings:
        indent = "  " * (lvl - 1)
        toc_lines.append(f"{indent}- [{txt}](#{anc})")

    toc_block = "\n".join(toc_lines) + "\n\n"
    content_body = "\n".join(new_lines).rstrip() + "\n"
    if not add_toc:
        return content_body
    # Xử lý front-matter để chèn TOC hợp lý
    if content_body.lstrip().startswith("---"):
        parts = re.split(r'^---\s*$', content_body, maxsplit=2, flags=re.MULTILINE)
        if len(parts) >= 3:
            final = parts[0] + "---" + parts[1] + "---\n\n" + toc_block + parts[2].lstrip()
        else: 
            final = toc_block + content_body
    else:
        final = toc_block + content_body

    # Path(output_path).write_text(final, encoding="utf-8")
    print(f"Xử lý xong! Đã tạo {len(headings)} mục heading.")
    return final

def adjust_page_markers(raw_text, 
                        # destination_path
                        ):
    """
    Reads a file, increments the integer found in '--- end of page=X ---',
    and writes to a new file.
    """
    # 2. Define the regex pattern for the page footer
    footer_pattern = r'--- end of page=(\d+) ---'

    # 3. Define the replacement logic
    def bump_index(match_obj):
        # Extract the digit string and convert to integer
        current_idx = int(match_obj.group(1))
        # Calculate the new value
        new_idx = current_idx + 1
        # Return the formatted replacement string
        return f'--- end of page={new_idx} ---'

    # 4. Process the text using regex substitution
    updated_text = re.sub(footer_pattern, bump_index, raw_text)

    # 5. Save the modified content to the destination file
    # with open(destination_path, 'w', encoding='utf-8') as dst:
    #     dst.write(updated_text)
    
    print(f"Done.")
    return updated_text

# ────────────────────────────────────────────────
# Part 4
# ────────────────────────────────────────────────
def split_markdown_to_leaf_sections(markdown_text, max_level_tag="###"):
    """
    Splits markdown into sections based on a maximum heading level.
    
    Args:
        markdown_text (str): The markdown content.
        max_level_tag (str): The header tag defining a 'leaf'. 
                             e.g., "###" means H3 is the split point.
                             H1 and H2 will be treated as context/parents.
                             H3 starts a new block.
                             H4+ are treated as content inside H3.
    """
    lines = markdown_text.splitlines()
    result = []
    
    # Calculate integer level from the tag string (e.g., "###" -> 3)
    target_level = len(max_level_tag.strip())
    
    # Dictionary to store current parent headers {1: "# A", 2: "## B"}
    context_headers = {}
    current_block = []
    
    def flush(force=False):
        nonlocal current_block
        if not current_block and not force:
            return
        
        # Build the path from context headers (Level 1 up to Target Level - 1)
        # We sort keys to ensure H1 comes before H2
        path = [context_headers[k] for k in sorted(context_headers.keys()) if k < target_level]
        
        if path or current_block:
            result.append("\n".join(path + current_block))
        
        current_block = []

    for line in lines:
        stripped = line.strip()
        
        # Handle empty lines (preserve them if we are inside a block)
        if not stripped:
            if current_block:
                current_block.append(line.rstrip())
            continue

        # Check if line is a header (starts with #)
        if stripped.startswith('#'):
            # Count the number of # to determine level
            level = 0
            for char in stripped:
                if char == '#':
                    level += 1
                else:
                    break
            
            # Ensure it is a valid markdown header
            is_header = (len(stripped) == level) or (len(stripped) > level and stripped[level] == ' ')
            
            if is_header:
                if level < target_level:
                    # It is a Context Header (Parent)
                    should_force = (level in context_headers)
                    flush(force=should_force)
                    
                    # Update context: Clear deeper levels, set current level
                    context_headers = {k: v for k, v in context_headers.items() if k < level}
                    context_headers[level] = line.rstrip()
                    current_block = []
                    
                elif level == target_level:
                    # It is a Leaf Header (The split point)
                    flush(force=False)
                    current_block = [line.rstrip()]
                    
                else:
                    # It is a Deeper Header (Content inside the leaf)
                    if context_headers or current_block or level > 1:
                         current_block.append(line.rstrip())
            else:
                current_block.append(line.rstrip())
        else:
            if context_headers or current_block:
                current_block.append(line.rstrip())

    # Final flush to capture trailing content/headers
    flush(force=True)
    return result

def clean_markdown_sections(sections):
    """
    Clean markdown sections by removing all leading # characters from each line
    and extra leading whitespace after the #s.
    
    Args:
        sections (list[str]): List of raw markdown section strings
        
    Returns:
        list[str]: List of cleaned section strings
    """
    cleaned_sections = []
    
    for text in sections:
        cleaned_lines = []
        for line in text.splitlines():
            # Remove all leading # characters
            stripped = line.lstrip('#')
            
            # If there's still leading space right after the #s, remove it
            if stripped and stripped[0].isspace():
                stripped = stripped.lstrip()
                
            cleaned_lines.append(stripped)
        
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_sections.append(cleaned_text)
    
    return cleaned_sections

def clean_markdown_sections_list(full_tag_md: str):
    # The input markdown from part 1)
    input_markdown = full_tag_md.strip()  # .strip() removes leading/trailing empty lines from the string

    # Run the split
    sections = split_markdown_to_leaf_sections(input_markdown, 
                                              max_level_tag="####")

    return clean_markdown_sections(sections)

def get_chunks(pdf_path: str):

    HEADING_RULES = [
        {"prefix": "chương",  "level": 2},
        {"prefix": "mục",     "level": 2},
        {"prefix": "phụ lục", "level": 2},
        {"prefix": "điều",    "level": 3},
        {"prefix": "digit",   "level": 5}, # 1., 2. -> H3
    ]

    converted_content = pdf_to_markdown(
        pdf_path     = pdf_path,
        output_md    = "tmp/converted.md",
        image_folder = "tmp/pdf_images",
        image_format = "png",          # or "jpeg" for smaller size
        # dpi          = 144,            # good balance quality/size
        remove_header = True,
        remove_footer = True,
        add_page_comments = True,
    )

    # print(converted_content)

    output_text = repair_markdown_file(converted_content)
    

    output_t = add_toc_to_markdown(output_text, add_toc=False, 
                                   heading_rules=HEADING_RULES)
    
    full_tag_md = adjust_page_markers(output_t)

    cleaned_sections = clean_markdown_sections_list(full_tag_md)

    print(f"Total sections extracted: {len(cleaned_sections)}")

    return cleaned_sections

if __name__ == "__main__":
    get_chunks("tests/test_data/2010.doc.pdf")
