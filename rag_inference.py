#!/usr/bin/env python3

import os
import sys
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rage_dataloader import RAGEDataLoader
from llm_router import LLMRouter

class RAGInference:
    """
    Coordinates ingestion, FAISS indexing, retrieval, and final generation.
    """
    def __init__(self,
                 data_folder='docs',
                 index_path='faiss_index',
                 embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                 chunk_size=128,
                 llm_backend='local'):
        self.data_folder = data_folder
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.llm_backend = llm_backend

        self.sbert = SentenceTransformer(embedding_model)
        self.llm_router = LLMRouter()
        self.text_data = []
        self.index = None

    def build_or_load_index(self):
        """
        Load existing FAISS index if found; otherwise build from folder.
        """
        if (os.path.exists(self.index_path + ".index") and
            os.path.exists(self.index_path + "_chunks.npy")):
            self.index = faiss.read_index(self.index_path + ".index")
            self.text_data = np.load(self.index_path + "_chunks.npy", allow_pickle=True)
        else:
            self._build_index_from_folder(self.data_folder)
            faiss.write_index(self.index, self.index_path + ".index")
            np.save(self.index_path + "_chunks.npy", self.text_data, allow_pickle=True)

    def _build_index_from_folder(self, folder):
        loader = RAGEDataLoader(chunk_size=self.chunk_size)
        all_chunks = loader.load_from_folder(folder)

        self.text_data = np.array(all_chunks, dtype=object)
        embeddings = self.sbert.encode(self.text_data.tolist()).astype(np.float32)

        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

    def ingest_data(self, filepaths=None, folderpaths=None, urls=None, chunk_size=None):
        """
        Ingest new data (files, folders, URLs), optionally override chunk_size.
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size

        if self.index is None:
            self.build_or_load_index()

        loader = RAGEDataLoader(chunk_size=self.chunk_size)
        new_chunks = []

        if filepaths:
            for fp in filepaths:
                new_chunks.extend(loader.load_file(fp))

        if folderpaths:
            for folder in folderpaths:
                new_chunks.extend(loader.load_from_folder(folder))

        if urls:
            for url in urls:
                new_chunks.extend(loader.load_from_url(url))

        combined = list(self.text_data) + new_chunks
        self.text_data = np.array(combined, dtype=object)

        embeddings = self.sbert.encode(self.text_data.tolist()).astype(np.float32)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

        faiss.write_index(self.index, self.index_path + ".index")
        np.save(self.index_path + "_chunks.npy", self.text_data, allow_pickle=True)

    def retrieve_context(self, query, top_k=3):
        if self.index is None:
            self.build_or_load_index()
        query_emb = self.sbert.encode([query]).astype(np.float32)
        distances, indices = self.index.search(query_emb, top_k)
        retrieved_chunks = [self.text_data[idx] for idx in indices[0]]
        return "\n".join(retrieved_chunks)

    def generate_response(self, query):
        context = self.retrieve_context(query, top_k=3)
        prompt = f"{query}\nRelevant context:\n{context}"
        model_output = self.llm_router.generate(prompt, backend=self.llm_backend)
        return {
            "retrieved_context": context,
            "final_response": model_output
        }

def main():
    """
    If first arg is 'ingest', read ingestion JSON from stdin (including chunk_size).
    Otherwise, normal RAG inference with query + backend in command line args.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        payload = json.loads(sys.stdin.read())
        chunk_size = payload.get('chunk_size', 128)

        rag = RAGInference(chunk_size=chunk_size)
        rag.build_or_load_index()

        rag.ingest_data(
            filepaths=payload.get("filepaths", []),
            folderpaths=payload.get("folderpaths", []),
            urls=payload.get("urls", []),
            chunk_size=chunk_size
        )
        print(json.dumps({"status": "success", "message": "Data ingested"}))

    else:
        user_query = sys.argv[1] if len(sys.argv) > 1 else "Test query"
        llm_backend = sys.argv[2] if len(sys.argv) > 2 else "local"
        rag = RAGInference(llm_backend=llm_backend)
        rag.build_or_load_index()
        result = rag.generate_response(user_query)
        print(json.dumps(result))

if __name__ == "__main__":
    main()
