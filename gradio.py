# !pip install -q transformers torch pillow groq python-dotenv pymupdf python-docx pdf2image numpy openai==0.28 gradio
# !apt-get update && apt-get install -y poppler-utils
#Florence- model link -https://huggingface.co/microsoft/Florence-2-large
import os
import gradio as gr
import torch
import tempfile
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from groq import Groq
import mimetypes
import openai
import numpy as np
from dotenv import load_dotenv


HF_TOKEN = os.getenv("HF_TOKEN")

try:
    import fitz
    import docx
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Document type detection
def detect_document_type(text_content):
    text_lower = text_content.lower()


    if any(term in text_lower for term in ["rx", "prescription", "sig:", "refill", "dispense", "mg"]):
        return "prescription"

    elif any(term in text_lower for term in ["lab", "test results", "reference range", "specimen"]):
        return "lab_report"

    elif any(term in text_lower for term in ["history", "assessment", "diagnosis", "plan", "soap", "progress note", "doctor note", "clinical note"]):
        return "medical_notes"

    elif any(term in text_lower for term in ["radiology", "x-ray", "mri", "ct scan", "ultrasound", "imaging"]):
        return "radiology"

    elif any(term in text_lower for term in ["insurance", "claim", "policy", "coverage", "authorization", "benefits", "subscriber", "insured", "copay", "deductible"]):
        return "insurance_claim"

    elif any(term in text_lower for term in ["discharge", "summary", "follow-up", "followup", "admission", "hospital course"]):
        return "discharge_summary"


    elif any(term in text_lower for term in ["pathology", "biopsy", "histology", "cytology", "specimen", "tissue"]):
        return "pathology_report"

# Default generic document
    else:
        return "generic"

# Document prompts
ENHANCEMENT_PROMPTS = {
    "prescription": """You are a medical prescription expert. Correct OCR errors in medicine names, dosages and medical terms. Preserve numbers/dates. Format with in markdown format with clear sections for:
1. Patient Info
2. Medications
3. Instructions
4. Doctor Info

IMPORTANT: Always include medication dosages on the SAME LINE as the medication name (e.g., 'Medication Name - dosage' or 'Medication Name dosage'). Do NOT list dosages on separate lines. If a medication does not have dosage information, SKIP listing that medication entirely.

Do NOT include any annotations about corrections. Do NOT show what words were corrected or include text in parentheses like '(corrected to:)'. Only provide the final corrected text without any indication of what was changed.""",

    "lab_report": """You are a medical laboratory report expert. Correct OCR errors in test names, values, and medical terminology. Preserve all numbers, units, and reference ranges exactly as they appear. Format with in markdown format with clear sections for:
2. Test Results (with values and reference ranges)
3. Interpretation/Notes
4. Laboratory Information

Maintain the tabular structure of test results where possible. Do NOT include any annotations about corrections.""",

    "medical_notes": """You are a medical documentation expert. Correct OCR errors in medical terminology, diagnoses, and treatment plans. Preserve the original structure of the notes. Format with clear sections for:
1. Patient Information
2. History
3. Assessment
4. Plan

Maintain any bullet points or numbered lists. Preserve all medical codes (ICD, CPT, etc.) exactly as they appear. Do NOT include any annotations about corrections.""",

    "radiology": """You are a radiology report expert. Correct OCR errors in anatomical terms, findings, and medical terminology. Preserve the original structure of the report. Format with clear sections for:
1. Patient Information
2. Examination Type
3. Findings
4. Impression/Conclusion

Maintain any measurements exactly as they appear. Do NOT include any annotations about corrections.""",

    "insurance_claim": """You are an insurance claim form expert. Correct OCR errors in policy numbers, procedure codes, diagnosis codes, and medical terminology. Preserve all numbers, dates, and monetary values exactly as they appear. Format with clear sections for:
1. Patient/Subscriber Information
2. Insurance Provider Information
3. Service Details
4. Diagnosis and Procedure Codes
5. Charges and Payment Information

Maintain the form structure and alignment where possible. Preserve all insurance codes (CPT, ICD, HCPCS) exactly as they appear. Do NOT include any annotations about corrections.""",

    "discharge_summary": """You are a hospital discharge summary expert. Correct OCR errors in medical terminology, medications, and treatment plans. Preserve the original structure of the document. Format with clear sections for:
1. Patient Information
2. Admission/Discharge Dates
3. Diagnoses
4. Hospital Course
5. Discharge Medications
6. Follow-up Instructions

Maintain any bullet points or numbered lists. Preserve all medical codes and dates exactly as they appear. Do NOT include any annotations about corrections.""",

    "referral_letter": """You are a medical referral letter expert. Correct OCR errors in medical terminology, diagnoses, and specialist information. Preserve the original structure of the letter. Format with clear sections for:
1. Referring Physician Information
2. Patient Information
3. Reason for Referral
4. Clinical History
5. Requested Consultation/Service

Maintain any letterhead formatting where possible. Preserve all contact information and dates exactly as they appear. Do NOT include any annotations about corrections.""",

    "pathology_report": """You are a pathology report expert. Correct OCR errors in anatomical terms, specimen descriptions, and diagnostic terminology. Preserve the original structure of the report. Format with clear sections for:
1. Patient Information
2. Specimen Details
3. Gross Description
4. Microscopic Examination
5. Diagnosis/Impression

Maintain any technical measurements and values exactly as they appear. Preserve all medical terminology and pathology codes precisely. Do NOT include any annotations about corrections.""",

    "generic": """You are a medical document expert. Correct OCR errors in medical terminology, names of conditions, treatments, and medications. Preserve all numbers, dates, and measurements exactly as they appear. Maintain the original document structure while improving readability. Do NOT include any annotations about corrections.
    If the document contains multiple pages separated by "--- New Page ---" markers, process each page separately. For each new page, start with a clear "## Page X" header.

     Do NOT include any annotations about corrections."""

}

def enhance_medical_text(text, doc_type=None):
    try:
        # Automatic detect
        if doc_type is None:
            doc_type = detect_document_type(text)

        # Prompt
        system_prompt = ENHANCEMENT_PROMPTS.get(doc_type, ENHANCEMENT_PROMPTS["generic"])

        if "--- New Page ---" in text:
            system_prompt += """

This document contains multiple pages separated by "--- New Page ---" markers.
For each page:
1. Process the content separately
2. Start each page with "## Page X" (where X is the page number)
3. Apply appropriate formatting for each page."""


        openai.api_key = "api_key"
        chat_completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Correct this medical document OCR output:\n{text}"
                }
            ],
            temperature=0.1,
            max_tokens=1024
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"OpenAI error: {str(e)}")
        return text



device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
HF_TOKEN = os.getenv("HF_TOKEN")
model = None
processor = None

# Load model
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        token=HF_TOKEN,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        token=HF_TOKEN,
        trust_remote_code=True
    )

    return model, processor

# Run OCR on an image
def run_ocr(image, model, processor, task_prompt="<OCR>"):
    # Prepare inputs for the model
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    # Generate output
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )

    # Decode the generated text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Convert PDF to images
def pdf_to_images(pdf_path):
    if not PDF_SUPPORT:
        raise ImportError("PDF support libraries not installed")

    try:
        return convert_from_path(pdf_path)
    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")
        return []

# Extract text from DOCX
def docx_to_text(docx_path):
    if not PDF_SUPPORT:  # We use the same flag for all document processing
        raise ImportError("DOCX support libraries not installed")

    try:
        doc = docx.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text from DOCX: {str(e)}")
        return ""

# Process for Gradio interface
def process_file_for_gradio(file_path):
    try:
        # Load model
        global model, processor
        if model is None or processor is None:
            model, processor = load_model()

        file_type = mimetypes.guess_type(file_path)[0]

        # Image file
        if file_type and file_type.startswith('image/'):
            image = Image.open(file_path).convert("RGB")
            ocr_text = run_ocr(image, model, processor)
            doc_type = detect_document_type(ocr_text)
            enhanced_text = enhance_medical_text(ocr_text, doc_type)

            return {
                "raw_text": ocr_text,
                "enhanced_text": enhanced_text,
                "document_type": doc_type
            }

        # PDF file
        elif file_type == 'application/pdf' and PDF_SUPPORT:
            images = pdf_to_images(file_path)
            if not images:
                return {"error": f"Could not extract images from PDF: {file_path}"}

            # Process one page for pdf
            all_text = []
            for img in images:
                page_text = run_ocr(img, model, processor)
                all_text.append(page_text)

            ocr_text = "\n\n--- New Page ---\n\n".join(all_text)
            doc_type = detect_document_type(ocr_text)
            enhanced_text = enhance_medical_text(ocr_text, doc_type)

            return {
                "raw_text": ocr_text,
                "enhanced_text": enhanced_text,
                "document_type": doc_type
            }

        # DOCX file
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' and PDF_SUPPORT:
            extracted_text = docx_to_text(file_path)
            if not extracted_text:
                return {"error": f"Could not extract text from DOCX: {file_path}"}

            ocr_text = extracted_text
            doc_type = detect_document_type(ocr_text)
            enhanced_text = enhance_medical_text(ocr_text, doc_type)

            return {
                "raw_text": ocr_text,
                "enhanced_text": enhanced_text,
                "document_type": doc_type
            }

        else:
            return {"error": f"Unsupported file type {file_type} for {file_path}"}

    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

# Gradio function
def process_and_display(file):
    if not file:
        return gr.Markdown("Please upload a file"), gr.Markdown(""), gr.Markdown("")

    result = process_file_for_gradio(file.name)

    if "error" in result:
        return gr.Markdown(f"Error: {result['error']}"), gr.Markdown(""), gr.Markdown("")

    doc_type = result["document_type"]
    raw_text = result["raw_text"]
    enhanced_text = result["enhanced_text"]

    return (
        gr.Markdown(f"**Document Type:** {doc_type}"),
        gr.Markdown(f"**Raw OCR Text:**\n```\n{raw_text}"),
        gr.Markdown(f"**Enhanced Text:**\n{enhanced_text}")
    )

# Create Gradio interface
with gr.Blocks(title="Medical Document OCR") as app:
    gr.Markdown("# Medical Document OCR ")
    gr.Markdown("Upload a medical document (image, PDF, or DOCX) to extract and enhance text using AI.")

    with gr.Row():
        file_input = gr.File(label="Upload Document")

    with gr.Row():
        process_button = gr.Button("Process Document")

    with gr.Row():
        doc_type_output = gr.Markdown()

    with gr.Tabs():
        with gr.TabItem("Raw OCR Text"):
            raw_text_output = gr.Markdown()
        with gr.TabItem("Enhanced Text"):
            enhanced_text_output = gr.Markdown()

    process_button.click(
        fn=process_and_display,
        inputs=file_input,
        outputs=[doc_type_output, raw_text_output, enhanced_text_output]
    )


if __name__ == "__main__":
    app.launch()