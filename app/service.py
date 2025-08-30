from itertools import product
from typing import TypedDict

import chromadb
import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LinearRegression


class SearchResult(TypedDict):
    document: str
    score: float
    embedding: list[float]


class TokenImportance(TypedDict):
    document: dict[str, float]
    query: dict[str, float]


def mean_pooling(
    token_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * mask_expanded).sum(0) / mask_expanded.sum(0)


class Service:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "xai",
        metric="l2",
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.metric = metric
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": metric, "ef_construction": 200}},
        )

    def dircet_token_importance(
        self,
        document_tokens: list[list[str]],
        document_attention_masks: list[torch.Tensor],
        document_token_embeddings: list[torch.Tensor],
        document_embeddings: NDArray[np.float32],
        query_tokens: list[str],
        query_attention_mask: torch.Tensor,
        query_token_embeddings: torch.Tensor,
        query_embedding: NDArray[np.float32],
        similarities: list[float],
    ) -> list[TokenImportance]:

        token_importance: list[TokenImportance] = list()

        assert (
            len(document_tokens)
            == len(document_attention_masks)
            == len(document_token_embeddings)
            == len(document_token_embeddings)
            == len(similarities)
        )

        for doc_tokens, doc_attn_mask, doc_token_embs, doc_embedding, full_sim in zip(
            document_tokens,
            document_attention_masks,
            document_token_embeddings,
            document_embeddings,
            similarities,
        ):

            # Documents -> Query
            document_contributions: dict[str, float] = dict()
            for i, tok in enumerate(doc_tokens):
                if tok in list(self.model.tokenizer.special_tokens_map.values()):
                    continue  # Ignore Special Tokens

                new_mask = doc_attn_mask.clone()
                new_mask[i] = 0  # Mask Token

                emb_sub = mean_pooling(
                    doc_token_embs, new_mask
                )  # Masked Document Embedding
                sim_sub = util.cos_sim(emb_sub, query_embedding)  # Masked Similarity
                document_contributions[tok] = sim_sub - full_sim  # Token Influence

            # Query -> Documents
            query_contributions: dict[str, float] = dict()
            for i, tok in enumerate(query_tokens):
                if tok in list(self.model.tokenizer.special_tokens_map.values()):
                    continue  # Ignore Special Tokens

                new_mask = query_attention_mask.clone()
                new_mask[i] = 0  # Mask Token

                emb_sub = mean_pooling(
                    query_token_embeddings, new_mask
                )  # Masked Query Embedding
                sim_sub = util.cos_sim(emb_sub, doc_embedding)  # Masked Similarity
                query_contributions[tok] = sim_sub - full_sim  # Token Influence

            token_importance.append(
                {"document": document_contributions, "query": query_contributions}
            )

        return token_importance

    def lime(
        self,
        document_tokens: list[list[str]],
        document_token_embeddings: list[torch.Tensor],
        document_embeddings: NDArray[np.float32],
        query_tokens: list[str],
        query_token_embeddings: torch.Tensor,
        query_embedding: NDArray[np.float32],
    ):

        assert (
            len(document_tokens)
            == len(document_token_embeddings)
            == len(document_token_embeddings)
        )

        lime_results = list()
        for doc_tokens, doc_tok_emb, doc_embs in zip(
            document_tokens, document_token_embeddings, document_token_embeddings
        ):

            # Document -> Query
            X, y = [], []

            tokens_core = doc_tokens[1:-1]  # remove CLS/SEP
            n_tokens = doc_tok_emb.size(0)
            n_core = len(tokens_core)

            # creating all binary (with or without token - 2^tokens) subsets (used as attenion mask for masking)
            for bits in product([0, 1], repeat=n_core):
                mask = torch.ones(n_tokens)  # Init Attention Mask
                mask[1:-1] = torch.tensor(bits)  # Mask Tokens

                emb_subset = mean_pooling(
                    doc_tok_emb, mask
                )  # Masked Document Embedding
                sim = util.cos_sim(emb_subset, query_embedding)

                X.append(mask.cpu().numpy())
                y.append(sim.squeeze().item())

            reg = LinearRegression().fit(X, y)
            coefs = reg.coef_

            # Query -> Doc

    def _unpack_search_results(self, search_results) -> list[SearchResult]:
        combined = list()
        combined = [
            {"document": doc, "score": score, "embedding": vector}
            for doc, score, vector in zip(
                search_results["documents"][0],
                search_results["distances"][0],
                search_results["embeddings"][0],
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
        q_emb = self.model.encode(query)
        search_results = self.collection.query(
            q_emb, n_results=top_k, include=["embeddings", "documents", "distances"]
        )
        results = self._unpack_search_results(search_results)

        # Document Embeddings
        documents = [r["document"] for r in results]
        doc_embeds = [r["embedding"] for r in results]
        doc_embeds = np.asarray(doc_embeds, dtype=np.float32)

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

        # Direct Token Importance
        token_importance = self.dircet_token_importance(
            document_tokens,
            document_attention_masks,
            document_token_embeddings,
            doc_embeds,
            query_tokens,
            query_attention_mask,
            query_token_embeddings,
            q_emb,
            full_similarities,
        )

        lime_res = self.lime(
            document_tokens,
            document_token_embeddings,
            doc_embeds,
            query_tokens,
            query_token_embeddings,
            q_emb,
        )

        return token_importance
