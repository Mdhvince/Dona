from langchain_text_splitters import RecursiveCharacterTextSplitter


def text_splitter(data, chunk_size, chunk_overlap):
    """
    Break down large documents into smaller, manageable pieces or chunks. This helps in processing and analyzing the
    text more efficiently, allowing the model to focus on specific sections rather than being overwhelmed by the entire
    document.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   length_function=len)
    chunks = text_splitter.split_documents(data)
    return chunks
