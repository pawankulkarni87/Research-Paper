import re

def extract_citations(text: str):
    """
    Extracts senetences with in-text citation like (Author,Year, Title) and 
    return surrounding context.
    """
    pattern = r"([^.]*?\([A-Z][a-z]+, \d{4}\)[^.]*\.)"
    matches = re.findall(pattern, text)
    contexts=[{"citation":match.strip(), "context": match.strip()} for match in matches]
    return contexts