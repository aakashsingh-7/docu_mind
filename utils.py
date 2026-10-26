def validate_pdf_upload(uploaded_file):
    """Basic validation for a Streamlit uploaded PDF."""
    if uploaded_file is None:
        raise ValueError("No file was provided.")
    name = getattr(uploaded_file, "name", "")
    if not name.lower().endswith(".pdf"):
        raise ValueError(f"{name or 'Uploaded file'} is not a PDF.")
    data = uploaded_file.getvalue()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"{name} does not appear to be a valid PDF file.")
    if len(data) == 0:
        raise ValueError(f"{name} is empty.")
