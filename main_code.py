

import os
import torch
import tempfile
import shutil
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import openai  
import mimetypes
import numpy as np
import glob
from IPython.display import display, Markdown


HF_TOKEN = "hugging_face_token"

try:
    import fitz
    import docx
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    print("PDF or DOCX support libraries not installed. Only image files will be processed.")
    PDF_SUPPORT = False

# Add the missing PDF functions
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
    if not PDF_SUPPORT:
        raise ImportError("DOCX support libraries not installed")

    try:
        doc = docx.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text from DOCX: {str(e)}")
        return ""

# Global model variables - initialize once
global_model = None
global_processor = None
def detect_document_type(text_content):
    text_lower = text_content.lower()

    if any(term in text_lower for term in ["rx", "prescription", "sig:", "refill", "dispense", "mg"]):
        return "prescription"

    elif any(term in text_lower for term in ["lab", "test results", "reference range", "specimen"]):
        return "lab_report"

    elif any(term in text_lower for term in ["radiology", "x-ray", "mri", "ct scan", "ultrasound", "imaging"]):
        return "radiology"

    elif any(term in text_lower for term in ["insurance", "claim", "policy", "coverage", "authorization", "benefits", "subscriber", "insured", "copay", "deductible"]):
        return "insurance_claim"

    elif any(term in text_lower for term in ["discharge", "summary", "follow-up", "followup", "admission", "hospital course"]):
        return "discharge_summary"

    elif any(term in text_lower for term in ["pathology", "biopsy", "histology", "cytology", "specimen", "tissue"]):
        return "pathology_report"
#default generic documnet
    else:
        return "generic"

# Document-specific enhancement prompts
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


    "pathology_report": """You are a pathology report expert. Correct OCR errors in anatomical terms, specimen descriptions, and diagnostic terminology. Preserve the original structure of the report. Format with clear sections for:
1. Patient Information
2. Specimen Details
3. Gross Description
4. Microscopic Examination
5. Diagnosis/Impression

Maintain any technical measurements and values exactly as they appear. Preserve all medical terminology and pathology codes precisely. Do NOT include any annotations about corrections.""",

    "generic": """You are a medical document expert. Correct OCR errors in medical terminology, names of conditions, treatments, and medications. Preserve all numbers, dates, and measurements exactly as they appear. Maintain the original document structure while improving readability. Do NOT include any annotations about corrections."""
}
#Singleton pattern to load model only once
def get_model():
    
    global global_model, global_processor
    
    if global_model is None or global_processor is None:
        print("Loading Florence-2-large model (first time initialization)")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        global_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            token=HF_TOKEN,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(device)

        global_processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large",
            token=HF_TOKEN,
            trust_remote_code=True
        )
    
    return global_model, global_processor, torch.cuda.is_available()



def enhance_medical_text(text, doc_type=None):
    try:
        if doc_type is None:
            doc_type = detect_document_type(text)

        system_prompt = ENHANCEMENT_PROMPTS.get(doc_type, ENHANCEMENT_PROMPTS["generic"])
        
        # Set API key directly
        openai_api_key = "api_key"
        openai.api_key = openai_api_key
        
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
        print(f"OpenAI enhancement error: {str(e)}")
        return text

# Run OCR on image
def run_ocr(image, model, processor, device_available, task_prompt="<OCR>"):
    # Get device and dtype based on availability
    device = "cuda:0" if device_available else "cpu"
    torch_dtype = torch.float16 if device_available else torch.float32
    
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



# Process file
def process_file(file_path, output_dir):
    print(f"Processing: {os.path.basename(file_path)}")
    file_type = mimetypes.guess_type(file_path)[0]
    
    # Get model (will only load if not already loaded)
    model, processor, device_available = get_model()

    # Image file
    if file_type and file_type.startswith('image/'):
        image = Image.open(file_path).convert("RGB")
        ocr_text = run_ocr(image, model, processor, device_available)
        doc_type = detect_document_type(ocr_text)
        enhanced_text = enhance_medical_text(ocr_text, doc_type)

    # PDF file
    elif file_type == 'application/pdf' and PDF_SUPPORT:
        images = pdf_to_images(file_path)
        if not images:
            print(f"Error: Could not extract images from PDF: {file_path}")
            return

        # Process each page
        all_text = []
        for img in images:
            page_text = run_ocr(img, model, processor, device_available)
            all_text.append(page_text)

        ocr_text = "\n\n- New Page -\n\n".join(all_text)
        doc_type = detect_document_type(ocr_text)
        enhanced_text = enhance_medical_text(ocr_text, doc_type)

    # DOCX file
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' and PDF_SUPPORT:
        extracted_text = docx_to_text(file_path)
        if not extracted_text:
            print(f"Error: Could not extract text from DOCX: {file_path}")
            return

        ocr_text = extracted_text
        doc_type = detect_document_type(ocr_text)
        enhanced_text = enhance_medical_text(ocr_text, doc_type)

    else:
        print(f"Error: Unsupported file type {file_type} for {file_path}")
        return

    # Print summary of results
    print(f"Detected document type: {doc_type}")
    print("\nRaw OCR Output :")
    print(ocr_text)
    print("\nEnhanced Medical Document")
    print(enhanced_text)

    # Save results to output dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.basename(os.path.splitext(file_path)[0])
        output_file = os.path.join(output_dir, f"{base_filename}_{doc_type}_result.txt")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(enhanced_text)
        print(f"Saved results to: {output_file}")

    return ocr_text, enhanced_text, doc_type

# Process files from a directory
def process_directory(input_dir, output_dir):
    # Create output dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Get all supported files from the directory
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    if PDF_SUPPORT:
        supported_extensions.extend(['.pdf', '.docx'])

    all_files = []
    for file in os.listdir(input_dir):
        ext = os.path.splitext(file)[1].lower()
        if ext in supported_extensions:
            all_files.append(os.path.join(input_dir, file))

    if not all_files:
        print("No supported files found in the specified directory.")
        return

    print(f"Found {len(all_files)} files to process.")

    # Process each file
    for file_path in all_files:
        try:
            process_file(file_path, output_dir)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

def main():
    # Set input and output directories directly in the code
    input_dir = "E:\\maincode\\ocr\\code\\custom\\input"
    output_dir = "E:\\maincode\\ocr\\code\\custom\\output"
    
    # Make sure paths are absolute
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    
    # Verify that input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist. Creating it now.")
        os.makedirs(input_dir)
    
    print(f"\nInput directory: {input_dir}")
    print(f"Results will be saved to: {output_dir}")

    # Process files
    print(f"\nProcessing files from {input_dir} to {output_dir}...\n")
    process_directory(input_dir, output_dir)
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()