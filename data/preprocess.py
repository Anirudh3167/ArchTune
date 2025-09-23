from datasets import Dataset
import re, os

def preprocess_wiki_text_generation(ds: Dataset) -> Dataset:
    """
    Preprocess Salesforce/wikitext-103 dataset.
    Returns a list of full articles as markdown strings.
    Each article:
      - Headings marked as ==, ===, ==== replaced by markdown #, ##, ### etc.
      - '@-@' replaced by '-', '@.@' replaced by '.'
      - <unk> tokens kept as is
    """
    # Compile regexes once outside the function for speed
    heading_pattern = re.compile(r"^(=+)\s*(.*?)\s*\1$")  # e.g., == Heading ==
    replacements = [
        (re.compile(r'@-\@'), '-'),
        (re.compile(r'@\s*-\s*@'), ' - '),
        (re.compile(r'@\s*\.\s*@'), '.'),
        (re.compile(r'@\s*,\s*@'), ','),
        (re.compile(r'@\s+@'), ' '),
    ]
    def clean_text(example):
            text = example["text"]
            # Heading pattern: == Heading ==, === Subheading ===, etc.
            match = heading_pattern.match(text)
            if match:
                level = len(match.group(1))  # e.g., '==' → 2
                heading_text = match.group(2).strip()
                markdown_heading = f"{'#' * level} {heading_text}"
                return {"text": markdown_heading}
            
            # Text pattern
            for pattern, repl in replacements:
                text = re.sub(pattern, repl, text)
            return {"text":text.strip()}
    ds = ds.map(clean_text, num_proc = max(1, os.cpu_count()-1))

    articles, curr_article = [], ""
    for example in ds:
        text: str = example["text"]
        if text.startswith("# "): # Level 1 Heading.
                articles.append({"text":curr_article + "\n\n"})
                curr_article = ""
        curr_article += text + "\n"
    if curr_article:
        articles.append({"text":curr_article})
    return Dataset.from_list(articles)

def preprocess_openwebtext(ds: Dataset) -> Dataset:
    """
    Preprocess vietgpt/openwebtext_en dataset.
    - Removes URLs
    - Cleans extra whitespace
    """

    # Regex patterns
    url_pattern = re.compile(r"https?://\S+|www\.\S+")

    def clean(example):
        text = example["text"]
        text = url_pattern.sub("", text)
        return {"text": text.strip()}

    ds = ds.map(clean, num_proc=max(1, os.cpu_count()-1))
    return ds
