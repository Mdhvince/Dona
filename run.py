import webapp
from webapp import app

if __name__ == "__main__":
    # use_reloader=False: the reloader would build the RAG pipeline (BM25
    # in-memory index included) twice and make the reindex thread fragile
    app.run(debug=True, use_reloader=False)
