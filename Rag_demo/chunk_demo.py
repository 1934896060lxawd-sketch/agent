from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1.读PDF
loader = PyPDFLoader("your_doc.pdf")
doc = loader.load()
text = "\n".join(d.page_content for d in doc)
print(f"原文档总长度:{len(text)} 字")

# 2.切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
)
chunks = splitter.split_text(text)

# 3. 看看切完的结果
print(f"切完共 {len(chunks)} 个 chunk\n")
print("第一个 chunk:")
print(chunks[0])
print("\n第二个 chunk(开头约 50 字和上一块重叠):")
print(chunks[1])

