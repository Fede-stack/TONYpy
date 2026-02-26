import numpy as np
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from dadapy import Data
from dadapy._utils import utils as ut
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import torch
import random
import warnings
import os
warnings.filterwarnings('ignore')


def set_global_seed(seed: int):
    """Globally set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class aRAG:
    """
    Adaptive Retrieval-Augmented Generation class using intrinsic dimensionality
    and dense retrievers for document selection.
    """

    def __init__(self, model_name='sentence-transformers/msmarco-MiniLM-L-12-v3',
                 k_fallback=5, random_seed=0):
        self.model_name = model_name
        self.k_fallback = k_fallback
        self.random_seed = random_seed  # FIX: era self.random_seed = random.seed(...) → None

        set_global_seed(random_seed)

        self.model = SentenceTransformer(model_name)
        self.rng = np.random.default_rng(random_seed)

        # Cache for document embeddings
        self.doc_embeddings = None
        self.documents = None

    def _reseed(self):
        """Re-apply seeds before any stochastic operation."""
        set_global_seed(self.random_seed)

    def _encode(self, texts, batch_size=32, show_progress_bar=False):
        """
        Deterministic encoding wrapper: re-seeds before every encode call.
        
        Using normalize_embeddings=True ensures cosine similarity is equivalent
        to dot product, removing any fp-order variance from normalization.
        """
        self._reseed()
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,  # deterministic normalization
        )

    def encode_documents(self, documents, batch_size=32, show_progress_bar=True):
        self.documents = documents
        self.doc_embeddings = self._encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

    def _compute_id_kstar_binomial(self, data, embeddings, initial_id=None,
                                   Dthr=23.92812698, r='opt', n_iter=10):
        self._reseed()  # dadapy uses np.random internally

        if initial_id is None:
            data.compute_id_2NN(algorithm='base')
        else:
            data.compute_distances()
            data.set_id(initial_id)

        ids = np.zeros(n_iter)
        kstars = np.zeros((n_iter, data.N), dtype=int)

        for i in range(n_iter):
            data.compute_kstar(Dthr)

            r_eff = min(0.95, 0.2032 ** (1. / data.intrinsic_dim)) if r == 'opt' else r

            rk = np.array([dd[data.kstar[j]] for j, dd in enumerate(data.distances)])
            rn = rk * r_eff
            n = np.sum([dd < rn[j] for j, dd in enumerate(data.distances)], axis=1)

            id_val = np.log((n.mean() - 1) / (data.kstar.mean() - 1)) / np.log(r_eff)

            data.set_id(id_val)
            ids[i] = id_val
            kstars[i] = data.kstar

        return ids, kstars[(n_iter - 1), :]

    def _find_k_neighbors(self, embeddings, query_index, k, use_cosine=True):
        target_embedding = embeddings[query_index]

        if use_cosine:
            all_distances = np.array([
                distance.cosine(target_embedding, emb) for emb in embeddings
            ])
            # FIX: kind='stable' avoids non-deterministic ordering of tied distances
            nearest_indices = np.argsort(all_distances, kind='stable')[1:k + 1]
        else:
            from sentence_transformers import util
            all_scores = util.dot_score(target_embedding, embeddings)[0].cpu().tolist()
            # FIX: kind='stable' on reversed array
            nearest_indices = np.argsort(all_scores, kind='stable')[::-1][1:k + 1]

        return nearest_indices.tolist()

    def _build_all_embeddings(self, query_embedding, doc_embeddings):
        """Concatenate query + doc embeddings in a reproducible way."""
        return np.concatenate(
            (np.array(query_embedding).reshape(1, -1), doc_embeddings)
        )

    def _adaptive_retrieve_single(self, query_embedding, documents, doc_embeddings,
                                   use_cosine, Dthr, r, n_iter):
        """Core retrieval logic for a single query — isolated for reuse."""
        if len(documents) <= self.k_fallback:
            return list(documents)

        try:
            all_embeddings = self._build_all_embeddings(query_embedding, doc_embeddings)

            self._reseed()  # seed before Data() instantiation
            data = Data(all_embeddings)

            ids, kstars = self._compute_id_kstar_binomial(
                data, doc_embeddings,
                initial_id=None, Dthr=Dthr, r=r, n_iter=n_iter
            )

            k_optimal = kstars[0]
            neighbor_indices = self._find_k_neighbors(all_embeddings, 0, k_optimal, use_cosine)
            doc_indices = np.array(neighbor_indices) - 1
            return np.array(documents)[doc_indices].tolist()

        except Exception as e:
            print(f"⚠️ Error in adaptive retrieval: {e}. Using fallback with k={self.k_fallback}")
            all_embeddings = self._build_all_embeddings(query_embedding, doc_embeddings)
            neighbor_indices = self._find_k_neighbors(
                all_embeddings, 0, min(self.k_fallback, len(documents)), use_cosine
            )
            doc_indices = np.array(neighbor_indices) - 1
            return np.array(documents)[doc_indices].tolist()

    def retrieve(self, query, documents=None, use_cosine=True, Dthr=23.92812698,
                 r='opt', n_iter=10):
        if documents is None:
            if self.doc_embeddings is None or self.documents is None:
                raise ValueError("No documents provided and no cached embeddings available.")
            documents = self.documents
            doc_embeddings = self.doc_embeddings
        else:
            doc_embeddings = self._encode(documents)

        query_embedding = self._encode(query)  # single string → 1-D array

        return self._adaptive_retrieve_single(
            query_embedding, documents, doc_embeddings, use_cosine, Dthr, r, n_iter
        )

    def retrieve_batch(self, queries, documents=None, use_cosine=True, Dthr=23.92812698,
                       r='opt', n_iter=10, batch_size=32, show_progress_bar=True):
        if documents is None:
            if self.doc_embeddings is None or self.documents is None:
                raise ValueError("No documents provided and no cached embeddings available.")
            documents = self.documents
            doc_embeddings = self.doc_embeddings
        else:
            doc_embeddings = self._encode(
                documents, batch_size=batch_size, show_progress_bar=show_progress_bar
            )

        query_embeddings = self._encode(
            queries, batch_size=batch_size, show_progress_bar=show_progress_bar
        )

        results = []
        for query_embedding in query_embeddings:
            results.append(
                self._adaptive_retrieve_single(
                    query_embedding, documents, doc_embeddings,
                    use_cosine, Dthr, r, n_iter
                )
            )

        return results

