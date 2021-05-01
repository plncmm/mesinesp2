import mesinesp2.data
import logging
logging.getLogger().setLevel(logging.INFO)

mesinesp2.data.download_annotated_data("data/raw")
mesinesp2.data.unzip_annotated_data("data/raw")