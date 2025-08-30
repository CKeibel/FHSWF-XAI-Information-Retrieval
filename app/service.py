from typing import TypedDict

import torch
from chroma import ChromaClient
from sentence_transformers import SentenceTransformer, util


class SearchResult(TypedDict):
    document: str
    score: float


class Service:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "xai",
        metric="l2",
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.metric = metric
        self.client = ChromaClient()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": metric, "ef_construction": 200}},
        )

    @staticmethod
    def mean_pooling(
        token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return (token_embeddings * mask_expanded).sum(0) / mask_expanded.sum(0)

    def dircet_token_importance():
        pass

    def _unpack_search_results(self, search_results) -> list[SearchResult]:
        combined = list()
        combined = [
            {"document": doc, "score": score}
            for doc, score in zip(
                search_results["documents"][0], search_results["distances"][0]
            )
        ]

        if self.metric == "cosine" or self.metric == "ip":
            # Highest score (similarity metric) on top
            return sorted(combined, key=lambda x: x["score"], reverse=True)
        else:
            # Lowest score (distance metric) on top
            return sorted(combined, key=lambda x: x["score"], reverse=False)

    def explain_embeddings(self, query: str, top_k: int = 5):  # TODO: Return Type
        # Query Collection
        q_emb = self.model.encode(query).tolist()
        search_results = self.collection.query(q_emb, n_results=top_k)
        results = self._unpack_search_results(search_results)

        # Document Embeddings
        documents = [r["document"] for r in results]
        doc_embeds = self.model.encode(documents)

        # Full Query to Documents Embeddings
        full_similarities = util.cos_sim(q_emb, doc_embeds).tolist()[0]

        # Calculate Document and Query properties for different xai methods
        encoded_query = self.model.tokenize([query])
        query_attention_mask = torch.tensor(encoded_query["attention_mask"])[0]
        query_tokens = self.model.tokenizer.convert_ids_to_tokens(
            encoded_query["input_ids"][0]
        )
        query_token_embeddings = (
            self.model.encode(
                query, output_value="token_embeddings", convert_to_tensor=True
            )
            .squeeze(0)
            .cpu()
        )

        encoded_documents = list()
        document_attention_masks = list()
        document_tokens = list()
        document_token_embeddings = list()

        for doc in documents:
            enc_doc = self.model.tokenize([doc])  # token ids + attention mask
            encoded_documents.append(enc_doc)
            doc_attn_mask = torch.tensor(enc_doc["attention_mask"])[0]
            document_attention_masks.append(doc_attn_mask)
            doc_tokens = self.model.tokenizer.convert_ids_to_tokens(
                enc_doc["input_ids"][0]
            )  # convert ids to tokens
            document_tokens.append(doc_tokens)
            doc_token_embs = (
                self.model.encode(
                    doc, output_value="token_embeddings", convert_to_tensor=True
                )
                .squeeze(0)
                .cpu()
            )
            document_token_embeddings.append(doc_token_embs)
