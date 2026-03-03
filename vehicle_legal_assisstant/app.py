import streamlit as st
from scripts.rag_pipeline import generate_answer
from pypdf import PdfReader

st.set_page_config(page_title="Motor Vehicle Legal Assistant", page_icon="🚦")
st.title(" Motor Vehicle Legal Assistant")
st.caption("Ask questions about Indian Motor Vehicle law — upload a challan PDF")

uploaded_file = st.file_uploader("Upload Traffic Challan (PDF)", type=["pdf"])

challan_text = ""
if uploaded_file:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        challan_text += page.extract_text() or ""
    if challan_text.strip():
        with st.expander("Challan text (extracted)"):
            st.text(challan_text[:2000])

question = st.text_input("Ask your question")

if st.button("Get Explanation"):
    
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching legal database..."):
            try:
                answer, sources = generate_answer(question, challan_text)
                st.write("### Answer")
                st.write(answer)

                # Show original source so anyone can verify the summary
                if sources and sources.strip():
                    with st.expander("📜 View original law text (source verification)"):
                        st.caption("This is the raw text from the Motor Vehicles Act that the answer is based on. "
                                   "You can compare it with the summary above to verify accuracy.")
                        st.text(sources)
            except Exception as e:
                st.error(f"Something went wrong: {e}")