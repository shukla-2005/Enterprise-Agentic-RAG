import os
import logfire

def parse_office(file_path: str) -> str:
    """
    Parses Office documents (.docx, .pptx) locally using python-docx and python-pptx.
    Falls back to Unstructured if available.
    """
    with logfire.span("Office Document Parsing", filename=file_path):
        try:
            ext = file_path.lower().rsplit(".", 1)[-1]
            if ext == "docx":
                import docx
                doc = docx.Document(file_path)
                paras = [p.text for p in doc.paragraphs if p.text.strip()]
                for table in doc.tables:
                    for row in table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                        if row_text:
                            paras.append(row_text)
                full_text = "\n\n".join(paras)
            elif ext == "pptx":
                from pptx import Presentation
                prs = Presentation(file_path)
                slides_text = []
                for slide in prs.slides:
                    slide_parts = [
                        shape.text.strip()
                        for shape in slide.shapes
                        if shape.has_text_frame and shape.text.strip()
                    ]
                    if slide_parts:
                        slides_text.append("\n".join(slide_parts))
                full_text = "\n\n".join(slides_text)
            else:
                from unstructured.partition.auto import partition
                elements = partition(filename=file_path)
                full_text = "\n".join([str(el) for el in elements])
            
            if not full_text.strip():
                logfire.warning(f"Empty text parsed for {file_path}")
            else:
                logfire.info(f"Successfully parsed {len(full_text)} characters from {file_path}")

            return full_text
        except Exception as e:
            logfire.error(f"Office Parse Failed for {file_path}: {e}")
            raise e
