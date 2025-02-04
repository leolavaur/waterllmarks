"""WaterLLMarks demo interface."""

import pandas as pd
import streamlit as st


def main():
    """Interface entry point."""
    st.sidebar.title("Settings")
    api_key = st.sidebar.text_input("API Key", type="password")
    watermark_type = st.sidebar.selectbox(
        "Watermark Type", ["Token", "Character Embedding"]
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("WaterLLMarks Chat Interface")
        chat_container = st.container()
        input_container = st.container()

        with chat_container:
            st.subheader("Chat Log")
            for log in st.session_state.logs:
                st.markdown(
                    f"<div style='text-align: right;'><b>You:</b> {log['Input']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='text-align: left;'><b>AI:</b> This is a mocked response.</div>",
                    unsafe_allow_html=True,
                )

        with input_container:
            chat_input = st.text_input("Type your message here...", key="chat_input")
            if st.button("Send"):
                # Placeholder for watermark application and validation
                st.session_state.logs.append({"Input": chat_input})
                st.rerun()

    with col2:
        st.title("Validation Logs")
        log_df = pd.DataFrame(st.session_state.logs)
        st.table(log_df)


if __name__ == "__main__":
    if "logs" not in st.session_state:
        st.session_state.logs = []
    main()
