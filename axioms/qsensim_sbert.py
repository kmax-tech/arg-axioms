
from ir_axioms.axiom import Axiom
from ir_axioms.axiom.utils import strictly_greater, approximately_equal
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from axioms.utils.socket_communication import _preference_vectors
import settings as s
class QSenSim_mean_sbert(Axiom):
    name = "QSenSim_mean_approxequal_sbert"

    def preference(self, context , query , document1 , document2) -> float:
        ranking_data = _preference_vectors(context=context,
                                           document1=document1,
                                           document2=document2,
                                           query=query,
                                           embedding_style=s.SBERT,
                                           comparison_method=s.MEAN,
                                           identifier=s.IDENT_SENTENCES

                                           )

        doc1_similarity = ranking_data[s.SOCKET_DOCUMENT1]
        doc2_similarity = ranking_data[s.SOCKET_DOCUMENT2]
        if approximately_equal(doc1_similarity,doc2_similarity):
            return 0

        return strictly_greater(
            doc1_similarity,
            doc2_similarity
        )

class QSenSim_mean_exact_sbert(Axiom):
    name = "QSenSim_mean_exact_sbert"

    def preference(self, context, query, document1, document2) -> float:
        ranking_data = _preference_vectors(context=context,
                                           document1=document1,
                                           document2=document2,
                                           query=query,
                                           embedding_style=s.SBERT,
                                           comparison_method=s.MEAN,
                                           identifier=s.IDENT_SENTENCES

                                           )
        doc1_similarity = ranking_data[s.SOCKET_DOCUMENT1]
        doc2_similarity = ranking_data[s.SOCKET_DOCUMENT2]
        return strictly_greater(
            doc1_similarity,
            doc2_similarity
        )

class QSenSim_max_sbert(Axiom):
    name = "QSenSim_max_approxequal_sbert"

    def preference(self, context, query, document1, document2) -> float:
        ranking_data = _preference_vectors(context=context,
                                           document1=document1,
                                           document2=document2,
                                           query=query,
                                           embedding_style=s.SBERT,
                                           comparison_method=s.MAX,
                                           identifier=s.IDENT_SENTENCES

                                           )
        doc1_similarity = ranking_data[s.SOCKET_DOCUMENT1]
        doc2_similarity = ranking_data[s.SOCKET_DOCUMENT2]
        if approximately_equal(doc1_similarity,doc2_similarity):
            return 0
        return strictly_greater(
            doc1_similarity,
            doc2_similarity
        )

class QSenSim_max_exact_sbert(Axiom):
    name = "QSenSim_max_exact_sbert"

    def preference(self, context, query, document1, document2) -> float:
        ranking_data  = _preference_vectors(context=context,
                                            document1=document1,
                                            document2=document2,
                                            query=query,
                                            embedding_style=s.SBERT,
                                            comparison_method=s.MAX,
                                            identifier=s.IDENT_SENTENCES

                                            )
        doc1_similarity = ranking_data[s.SOCKET_DOCUMENT1]
        doc2_similarity = ranking_data[s.SOCKET_DOCUMENT2]
        return strictly_greater(
            doc1_similarity,
            doc2_similarity
        )





