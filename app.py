import gradio as gr
import PyPDF2
from pdf2image import convert_from_path
import tempfile

from langchain_community.chat_models import ChatOllama


def extract_text_from_pdf(pdf_file):
    with open(pdf_file, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def preview_pdf_as_image(pdf_file):
    # Convertir la primera página del PDF a una imagen
    images = convert_from_path(pdf_file, first_page=0, last_page=1)
    # Guardar la imagen en un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
        images[0].save(temp_img.name, "PNG")
        return temp_img.name


def extract_categories(text):
    # Definir la función para extraer categorías
    functions = [
        {
            "name": "extract_categories",
            "description": "Extract the most important categories from the provided text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "A category extracted from the text.",
                        },
                    }
                },
                "required": ["categories"],
            },
        }
    ]

    llm = ChatOllama(model="llama3.1", temperature=0, functions=functions)

    # Enviar el texto al LLM para extraer las categorías
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]

    response = llm.invoke(messages)

    # Parsear la respuesta JSON y retornar
    return response.content


def process_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    categories = extract_categories(text)
    return categories


# Crear la interfaz de Gradio
with gr.Blocks() as demo:
    with gr.Row():
        pdf_input = gr.File(label="Cargar PDF", type="filepath")
        pdf_preview = gr.Image(label="Vista Previa del PDF", interactive=False)

    # Mostrar la vista previa cuando se cargue el archivo
    pdf_input.change(preview_pdf_as_image, pdf_input, pdf_preview)

    with gr.Row():
        extract_btn = gr.Button("Extraer Categorías")
        output_text = gr.Textbox(label="Categorías Extraídas", interactive=False)

    # Extraer y mostrar las categorías cuando se presione el botón
    extract_btn.click(process_pdf, pdf_input, output_text)
    # Iniciar la aplicación Gradio
demo.launch()
