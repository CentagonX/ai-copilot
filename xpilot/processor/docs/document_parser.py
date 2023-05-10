import os
import io
import json
import textract
import openai
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from PyPDF2 import PdfReader, PdfWriter
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

print (openai.organization)
print (openai.api_key)

root_path = os.getcwd()
sample_pdf_path = os.path.join(root_path, "docs/matter/a1n3c000009NF1TAAW/aDu3c000006iMsjCAE.pdf")

doc_path = urlparse(sample_pdf_path)
document_name = os.path.basename(doc_path.path) 

case_pdf = PdfReader(sample_pdf_path)
number_of_pages = len(case_pdf.pages)
print ("Number of pages: " + str(number_of_pages))

output = []
i = 0

for page in case_pdf.pages:

    page = case_pdf.pages[i]
    i = i + 1
    output.append(page.extract_text())

pages = []
j = 0
for e in output:
    value = " ".join(str(e))
    pages.append(value)
    j = j + 1

docs = dict({"document_id": document_name, "pages": pages})

sample_json_path = os.path.join(root_path, "docs/matter/a1n3c000009NF1TAAW/aDu3c000006iMsjCAE.json")
with open(sample_json_path, 'w') as sample_json_file_path:
    json.dump(docs, sample_json_file_path, indent=2)
