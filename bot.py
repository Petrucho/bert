import os
import logging
import nlp_lr
import streamlit as st
# from config import TOKEN

import asyncio

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)



# this part of code needed in Streamlit - never the mind whatever it's mean
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()
asyncio.set_event_loop(st.session_state.loop)


# if __name__ == '__main__':    