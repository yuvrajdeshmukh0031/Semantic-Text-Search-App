"""
data.py
-------
This file stores the sample dataset of sentences used for semantic search.
The sentences cover topics like Artificial Intelligence, Machine Learning,
Python Programming, Sports (Cricket, Football), and General Technology.
"""

# Sample dataset containing 18 sentences across different topics
SAMPLE_SENTENCES = [
    # Artificial Intelligence & Machine Learning
    "Artificial intelligence allows computers to learn from data and make smart decisions.",
    "Machine learning models learn patterns from training data to predict future outcomes.",
    "Deep learning is a subset of machine learning based on artificial neural networks.",
    "Natural language processing helps computers understand and process human language.",

    # Python Programming
    "Python is a versatile programming language widely used in data science and web development.",
    "Writing clean and readable Python code makes software maintenance easier.",
    "Pandas and NumPy are powerful Python libraries for data manipulation and analysis.",

    # Sports - Cricket
    "Cricket is a popular team sport played with a bat, ball, and wickets.",
    "The batsman hit a massive six over the deep mid-wicket boundary.",
    "Fast bowlers aim to take early wickets by swinging the cricket ball.",

    # Sports - Football
    "Football is the world's most popular sport, played by millions across the globe.",
    "The striker scored a stunning goal in the final minutes of the match.",
    "The goalkeeper made a crucial save to keep his team in the championship.",

    # Technology & General Computing
    "Cloud computing enables companies to store data and run applications on remote servers.",
    "Cybersecurity measures protect systems and networks from digital attacks.",
    "Quantum computing has the potential to solve complex mathematical problems fast.",
    "Smartphones have revolutionized the way people communicate and access information.",
    "Software engineers build applications to automate everyday user tasks."
]


def get_sample_sentences() -> list[str]:
    """
    Returns the list of sample sentences.
    
    Returns:
        list[str]: List of sentences for semantic search dataset.
    """
    return SAMPLE_SENTENCES


def extract_sentences_from_pdf(pdf_file) -> list[str]:
    """
    Extracts text from an uploaded PDF file and splits it into sentences.

    Args:
        pdf_file: Uploaded PDF file object (e.g., from Streamlit file_uploader).

    Returns:
        list[str]: Clean list of extracted sentences.
    """
    import re
    from pypdf import PdfReader

    try:
        reader = PdfReader(pdf_file)
        full_text = ""
        
        # Read text page by page
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + " "

        # Replace multiple spaces/newlines with a single space
        cleaned_text = re.sub(r'\s+', ' ', full_text).strip()

        if not cleaned_text:
            return []

        # Split into sentences using punctuation marks (. ! ?)
        raw_sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)

        # Retain sentences with a minimum length of 10 characters
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]
        return sentences
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []

