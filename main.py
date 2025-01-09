import os
from dotenv import load_dotenv
import streamlit as st

# models
from langchain_openai import ChatOpenAI

from urllib.parse import urlparse
import base64
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

load_dotenv()

PROMPT = """
まず、以下のユーザーのリクエストとアップロードされた画像を注意深く読んでください。
次に、アップロードされた画像に基づいて画像を生成するというユーザーのリクエストに
沿った DALL-E プロンプトを作成してください。
DALL-E プロンプトは必ず英語で作成してください。

ユーザー入力: {user_input}

プロンプトでは、ユーザーがアップロードした写真に何が描かれているか、どのように構成
されているかを詳細に説明してください。
写真に何が写っているのかはっきりと見える場合は、示されている場所や人物の名前を正確に
書き留めてください。
写真の構図とズームの程度を可能な限り詳しく説明してください。
写真の内容を可能な限り正確に再現することが重要です。

DALL-E 3 向けのプロンプトを英語で回答してください。
"""

def init_page():
    st.set_page_config(
        page_title="Image AI Recognizer",
        page_icon="🤖",
    )
    st.header("Image AI Recognizer 🤖")

def main():
    init_page()
    
    llm = ChatOpenAI(
        temperature=0,
        model="deepseek-chat",
        openai_api_key=os.getenv("DEEP_SEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com",
        max_tokens=512,
    )
    
    dalle3_image_url = None
    upload_file = st.file_uploader(
        label="Upload your Image here",
        type=["png", "jpg", "jpeg","webp","gif"],
    )
    if upload_file:
        if user_input := st.chat_input("画像をどのように加工したいか教えて下さい"):
            image_base64 = base64.b64encode(upload_file.read()).decode()
            image = f"data:image/jpeg;base64,{image_base64}"
            
            query = [
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": PROMPT.format(user_input=user_input),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image,
                                "detail": "auto"
                            }
                        }
                    ]
                )
            ]
            print(query)
    
            st.markdown("## Image Prompt")
            image_prompt = st.write_stream(llm.stream(query))
            
            with st.spinner("DALL-E 3 で画像生成中..."):
                dalle3 = DallEAPIWrapper(
                    model="dall-e-3",
                    size="1792x1024",
                    quality="standard",
                    n=1
                )
                dalle3_image_url = dalle3.run(image_prompt)
    else:
        st.write("まずは画像をアップロードしてください")
        
    if dalle3_image_url:
        st.markdown("### Question")
        st.write(user_input)
        st.image(
            upload_file,
            use_column_width="auto",
        )
        st.markdown("### DALL-E 3 で生成した画像")
        st.image(
            dalle3_image_url,
            caption=image_prompt,
            use_column_width="auto",
        )

if __name__ == "__main__":
    main()