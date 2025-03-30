import os
import getpass
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

# Configuration
PDF_PATH = "PDF_PATH"  # Path to the PDF file
VECTOR_DB_PATH = "DB_PATH"  # Path to save the FAISS vector database
MODEL_ID = "meta-llama/Llama-2-13b-chat-hf"  # LLM model ID
EMBEDDING_MODEL = "BAAI/bge-m3"  # Embedding model name

def setup_environment():
    """Set up environment variables for API keys"""
    if not os.getenv("UNSTRUCTURED_API_KEY"):
        os.environ["UNSTRUCTURED_API_KEY"] = getpass.getpass("UNSTRUCTURED_API_KEY")

    if not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = getpass.getpass("HF_TOKEN")

    print("Unstructured API Key installed:", os.getenv("UNSTRUCTURED_API_KEY") is not None)
    print("HF API Key installed:", os.getenv("HF_TOKEN") is not None)

def load_and_process_pdf(file_path):
    """Load PDF and split into chunks"""
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    try:
        print(f"Đang xử lý: {os.path.basename(file_path)}")
        loader = UnstructuredPDFLoader(file_path, mode="elements", strategy="fast")
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
        print(f"Số lượng documents đã tải: {len(documents)}")
        print(f"Số lượng chunks đã tạo: {len(chunks)}")
        del documents
        del chunks
    except Exception as e:
        print(f"Lỗi khi đang xử lý {file_path}: {e}")

    # Remove duplicate chunks
    unique_text = {}
    chunk_processed_unique = []
    for chunk in all_chunks:
        if chunk.page_content not in unique_text:
            unique_text[chunk.page_content] = 1
            chunk_processed_unique.append(chunk)

    del unique_text
    print(f"Tổng số chunk sau khi loại bỏ trùng lặp: {len(chunk_processed_unique)}")
    
    return chunk_processed_unique

def create_vector_store(chunks):
    """Create and save vector embeddings"""
    model_kwargs = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'trust_remote_code': True
    }

    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_DB_PATH)
    
    return vector_store

def setup_llm():
    """Initialize the LLM model"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.65,
        top_p=0.9,
        repetition_penalty=1.03,
        do_sample=True
    )

    return HuggingFacePipeline(pipeline=llm_pipeline)

def create_qa_chain(vector_store, llm):
    """Create the question-answering chain"""
    template = """<s>[INST] <<SYS>>
Bạn là trợ lý AI tiếng Việt hữu ích. Hãy sử dụng ngữ cảnh sau đây để trả lời câu hỏi của người dùng bằng tiếng Việt.
Ngữ cảnh: {context}
<</SYS>>

Câu hỏi: {question}
Trả lời: [/INST]
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3},
        search_type="similarity",
        distance_strategy=DistanceStrategy.COSINE
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )
    
    return qa_chain

def test_sample_questions(qa_chain):
    """Test the QA system with sample questions"""
    cau_hoi_mau = [
        "Thời gian thẩm định báo cáo kinh tế kỹ thuật?",
        "Tóm tắt nội dung chính Nghị định 175 về báo cáo tiền khả thi"
    ]

    for cau_hoi in cau_hoi_mau:
        print(f"\nCâu hỏi: {cau_hoi}")
        tra_loi = qa_chain.run(cau_hoi)
        print(f"Trả lời: {tra_loi}")

def interactive_qa(qa_chain):
    """Run an interactive Q&A session"""
    while True:
        cau_hoi_nguoi_dung = input("Nhập câu hỏi (hoặc 'thoat' để dừng): ")
        if cau_hoi_nguoi_dung.lower() == "thoat":
            break
        
        tra_loi = qa_chain.run(cau_hoi_nguoi_dung)
        print(f"Câu hỏi: {cau_hoi_nguoi_dung}")
        print(f"Trả lời: {tra_loi}")

def main():
    """Main function to run the entire pipeline"""
    print("=== Setting up environment ===")
    setup_environment()
    
    print("\n=== Processing PDF ===")
    chunks = load_and_process_pdf(PDF_PATH)
    
    print("\n=== Creating Vector Store ===")
    vector_store = create_vector_store(chunks)
    
    print("\n=== Setting up LLM ===")
    llm = setup_llm()
    
    print("\n=== Creating QA Chain ===")
    qa_chain = create_qa_chain(vector_store, llm)
    
    print("\n=== Testing Sample Questions ===")
    test_sample_questions(qa_chain)
    
    print("\n=== Starting Interactive Q&A ===")
    interactive_qa(qa_chain)

if __name__ == "__main__":
    main()