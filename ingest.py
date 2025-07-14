import os
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from constants import URL

class DocumentIngestor:
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separator="\n"
        )
    
    def load_pdfs(self, pdf_directory="data/pdfs/"):
        documents = []
        
        if not os.path.exists(pdf_directory):
            print(f"Directory {pdf_directory} not found")
            return documents
            
        for filename in os.listdir(pdf_directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, filename)
                try:
                    reader = PdfReader(pdf_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": filename, "type": "pdf"}
                    ))
                    print(f"Loaded PDF: {filename}")
                except Exception as e:
                    print(f"Error loading PDF {filename}: {e}")
        
        return documents
    
    def load_docx_files(self, directory="data/pdfs/"):
        documents = []
        
        if not os.path.exists(directory):
            print(f"Directory {directory} not found")
            return documents
            
        for filename in os.listdir(directory):
            if filename.endswith('.docx') and not filename.startswith('~$'):  # Skip temp files
                docx_path = os.path.join(directory, filename)
                try:
                    doc = DocxDocument(docx_path)
                    
                    text_content = []
                    for paragraph in doc.paragraphs:
                        if paragraph.text.strip():
                            text_content.append(paragraph.text.strip())
                    
                    for table in doc.tables:
                        for row in table.rows:
                            row_text = []
                            for cell in row.cells:
                                if cell.text.strip():
                                    row_text.append(cell.text.strip())
                            if row_text:
                                text_content.append(" | ".join(row_text))
                    
                    full_text = "\n".join(text_content)
                    
                    if full_text.strip():
                        documents.append(Document(
                            page_content=full_text,
                            metadata={"source": filename, "type": "docx"}
                        ))
                        print(f"Loaded DOCX: {filename}")
                    else:
                        print(f"Warning: DOCX file {filename} appears to be empty")
                        
                except Exception as e:
                    print(f"Error loading DOCX {filename}: {e}")
        
        return documents
    
    def load_all_documents(self, directory="data/pdfs/"):
        documents = []
        
        pdf_docs = self.load_pdfs(directory)
        documents.extend(pdf_docs)
        
        docx_docs = self.load_docx_files(directory)
        documents.extend(docx_docs)
        
        print(f"Total documents loaded: {len(documents)} ({len(pdf_docs)} PDFs, {len(docx_docs)} DOCX)")
        return documents
    
    def scrape_webpage(self, url=URL):
        documents = []
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.main-content', 'section', 'div.container'
            ]
            
            content_found = False
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        text = element.get_text(separator='\n', strip=True)
                        if len(text) > 100:
                            documents.append(Document(
                                page_content=text,
                                metadata={"source": url, "type": "webpage"}
                            ))
                            content_found = True
                            break
                if content_found:
                    break
            
            if not content_found:
                text = soup.get_text(separator='\n', strip=True)
                if len(text) > 100:
                    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
                    for i, chunk in enumerate(chunks):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={"source": f"{url}#chunk{i+1}", "type": "webpage"}
                        ))
            
            print(f"Scraped {len(documents)} sections from {url}")
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            documents.append(Document(
                page_content="Unable to scrape webpage content. Please check the URL or try again later.",
                metadata={"source": url, "type": "webpage", "error": str(e)}
            ))
        
        return documents
    
    def process_documents(self, documents):
        if not documents:
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

if __name__ == "__main__":
    ingestor = DocumentIngestor()
    
    all_file_docs = ingestor.load_all_documents()
    
    web_docs = ingestor.scrape_webpage(URL)
    
    all_docs = all_file_docs + web_docs
    
    # Process into chunks
    chunks = ingestor.process_documents(all_docs)
    
    print(f"Total chunks created: {len(chunks)}")