# Web Scraping, Processing, and Embedding

This Google Colab notebook demonstrates a complete workflow for extracting information from a website, processing it, and preparing it for use in applications like question answering or semantic search.

Here's a breakdown of its functionality:

1.  **Web Scraping:** It starts by crawling a specified website (in this case, `learn.microsoft.com/en-us/`) and extracting the text content from multiple pages.
2.  **Data Processing:** The extracted text is then processed by splitting it into smaller, overlapping chunks. This makes the text more manageable for further analysis and embedding.
3.  **Embedding:** Using a pre-trained Sentence Transformer model, each text chunk is converted into a numerical representation called an embedding. Embeddings capture the semantic meaning of the text, allowing for comparisons of similarity between different chunks.
4.  **Saving:** Finally, the processed data, including the original text chunks, their source URLs, and the newly created embeddings, is saved to a JSON file in your Google Drive for later use.

In essence, the notebook takes raw website text and transforms it into a structured, semantically rich dataset.

## Setup

1.  Open the provided Google Colab notebook.
2.  Install the necessary libraries by running the first code cell:

bash !pip install -q ipywidgets google-colab python-docx pypdf pandas nltk sentence-transformers torch tqdm pyarrow httpx beautifulsoup4 datasets requests

3.  Mount your Google Drive to save the output:

python from google.colab import drive drive.mount('/content/drive')

## Usage

1.  **Web Scraping:** Run the second code cell to start the web scraping process. The script will crawl the specified `start_url` and collect text data from the linked pages within the same domain. You can adjust `max_pages` in the `crawl_website` function to control the number of pages scraped.
2.  **Data Processing and Embedding:** Run the third code cell to process the scraped data into chunks and generate embeddings using the 'all-MiniLM-L6-v2' Sentence Transformer model.
3.  **Save Embedded Data:** Run the fourth code cell to save the processed and embedded data to a JSON file named `embedded_dataset.json` in the `Output` folder of your Google Drive.

## Output

The notebook will generate an `embedded_dataset.json` file in your Google Drive containing a list of dictionaries. Each dictionary represents a text chunk and includes:

*   `text`: The text content of the chunk.
*   `source`: The URL from which the chunk was extracted.
*   `embedding`: A list of floating-point numbers representing the embedding of the text chunk.

This dataset can be used for various downstream tasks such as building a question-answering system or implementing semantic search.

## Customization

*   **Start URL:** Modify the `start_url` variable in the web scraping script to crawl a different website.
*   **Chunking Parameters:** Adjust `chunk_size` and `chunk_overlap` in the `chunk_text` function to control how the text is split.
*   **Sentence Transformer Model:** Change the model name in the `SentenceTransformer` constructor to use a different pre-trained model.
*   **Output Path:** Modify the `output_file_path` variable to save the output file to a different location in your Google Drive.
