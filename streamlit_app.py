import streamlit as st
import main
#import validation.validate as val

def run_main():
    main.main_qa()

def run_validation():
    # ***** Uncomment the below line *******
    #validation_result = val.get_validation_result()
    validation_result={'context_precision': 0.6389, 'faithfulness': 1.0000, 'answer_relevancy': 0.9780, 'context_recall': 1.0000}
    with st.container(border=True):
        st.header("Validation Results")
        st.divider()
        st.markdown("Retriver Performance: ")
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("context_precision")
            st.divider()
            st.write(validation_result['context_precision'])
        with col2:
            st.markdown("context_recall")
            st.divider()
            st.write(validation_result['context_recall'])
        st.divider()
        st.markdown("Generator Performance: ")
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("faithfulness")
            st.divider()
            st.write(validation_result['faithfulness'])
        with col2:
            st.markdown("answer_relevancy")
            st.divider()
            st.write(validation_result['answer_relevancy']) 
 
tab1, tab2 = st.tabs(["RAG based Q & A", "Validation Results"])

with tab1:
    run_main()

with tab2:
    run_validation()

