import base64
import io

from langchain_core.messages import HumanMessage

RENDER_DPI = 150


def llm_bases_img2text(vlm, image_bytes, mime, prompt):
    """
    This function converts an image to text using a vision language model (VLM).
    :param vlm: The vision language model to use for the conversion.
    :param image_bytes: The image data in bytes format.
    :param mime: The MIME type of the image (e.g., "image/png", "image/jpeg").
    :param prompt: The prompt to guide the VLM in the conversion process.
    :return: The text output from the VLM after processing the image.
    """
    b64 = base64.b64encode(image_bytes).decode()
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ])
    return vlm.invoke([message]).content


def pdf2png(page, dpi=RENDER_DPI):
    """
    Render a PDF page to PNG bytes at the given resolution
    :param page: The PDF page to render.
    :param dpi: The resolution in dots per inch (default is 150).
    :return: The rendered PNG image in bytes format.
    """
    image = page.render(scale=dpi / 72).to_pil()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def iter_files(roots, suffixes):
    """
    Yield (root, path) for every ingestable file, skipping private folders (any path component starting with "_").
    :param roots: A list of root directories to search for files.
    :param suffixes: A set of file suffixes to include.
    :return: A generator yielding tuples of (root, path)
    """
    for root in roots:
        if not root.exists():
            print(f"Chemin introuvable, ignorée : {root}")
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in suffixes:
                print(f"\t\t{path.name} : suffixe ignoré", flush=True)
                continue
            if any(part.startswith("_") for part in path.relative_to(root).parts[:-1]):
                continue
            yield root, path
