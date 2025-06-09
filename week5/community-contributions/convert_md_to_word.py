import os
import pypandoc

# Try to download pandoc if it's not found
pypandoc.download_pandoc()

def convert_md_to_docx_in_subfolder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".md"):
                md_path = os.path.join(root, file)
                docx_path = os.path.splitext(md_path)[0] + ".docx"
                try:
                    pypandoc.convert_file(md_path, 'docx', outputfile=docx_path)
                    print(f"✅ Converted: {md_path} -> {docx_path}")
                except Exception as e:
                    print(f"❌ Failed to convert {md_path}: {e}")

# Example usage
convert_md_to_docx_in_subfolder("./my_llm_knowledge")
