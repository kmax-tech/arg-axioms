from ir_axioms.axiom import Axiom
from ir_axioms.axiom.utils import strictly_greater, approximately_equal
from axioms.utils.socket_communication import _preference_vectors
import settings as s

class QArgSim_mean_exact_sbert(Axiom):
    name = "QArgSim_mean_exact_sbert_targer"

    def preference(self, context, query, document1, document2) -> float:
        ranking_data = _preference_vectors(context=context,
                                           document1=document1,
                                           document2=document2,
                                           query=query,
                                           embedding_style=s.SBERT,
                                           comparison_method=s.MEAN,
                                           identifier= s.IDENT_AUS_SINGLE_SENTENCE
                                           )
        doc1_similarity = ranking_data[s.SOCKET_DOCUMENT1]
        doc2_similarity = ranking_data[s.SOCKET_DOCUMENT2]
        return strictly_greater(
            doc1_similarity,
            doc2_similarity
        )



class QArgSim_mean_exact_sbert_full_document(Axiom):
    name = "QArgSim_mean_exact_sbert_targer_full_document"

    def preference(self, context, query, document1, document2) -> float:
        ranking_data = _preference_vectors(context=context,
                                           document1=document1,
                                           document2=document2,
                                           query=query,
                                           embedding_style=s.SBERT,
                                           comparison_method=s.MEAN,
                                           document_sentenice= False,
                                           identifier= s.IDENT_AUS_FULL_DOCUMENT,
                                           task_info= {s.TASK_INFO_TARGER_OWN_SENTENCIZER : True}
                                           )
        doc1_similarity = ranking_data[s.SOCKET_DOCUMENT1]
        doc2_similarity = ranking_data[s.SOCKET_DOCUMENT2]
        return strictly_greater(
            doc1_similarity,
            doc2_similarity
        )
    
class QArgSim_mean_sbert(Axiom):
    name = "QArgSim_mean_approxequal_sbert"

    def preference(self, context, query, document1, document2) -> float:
        ranking_data = _preference_vectors(context=context ,
                                           document1=document1 ,
                                           document2=document2 ,
                                           query=query ,
                                           embedding_style=s.SBERT ,
                                           comparison_method=s.MEAN ,
                                           identifier=s.IDENT_AUS_SINGLE_SENTENCE

                                           )
        doc1_similarity = ranking_data[s.SOCKET_DOCUMENT1]
        doc2_similarity = ranking_data[s.SOCKET_DOCUMENT2]
        if approximately_equal(doc1_similarity,doc2_similarity):
            return 0
        return strictly_greater(
            doc1_similarity,
            doc2_similarity
        )


class QArgSim_mean_sbert_full_document(Axiom):
    name = "QArgSim_mean_approxequal_sbert_full_document"

    def preference(self, context, query, document1, document2) -> float:
        ranking_data = _preference_vectors(context=context,
                                           document1=document1,
                                           document2=document2,
                                           query=query,
                                           embedding_style=s.SBERT,
                                           comparison_method=s.MEAN,
                                           document_sentenice=False,
                                           identifier=s.IDENT_AUS_FULL_DOCUMENT,
                                           task_info={s.TASK_INFO_TARGER_OWN_SENTENCIZER : True}

                                           )
        doc1_similarity = ranking_data[s.SOCKET_DOCUMENT1]
        doc2_similarity = ranking_data[s.SOCKET_DOCUMENT2]
        if approximately_equal(doc1_similarity,doc2_similarity):
            return 0
        return strictly_greater(
            doc1_similarity,
            doc2_similarity
        )

class QArgSim_max_exact_sbert(Axiom):
    name = "QArgSim_max_exact_sbert"

    def preference(self, context, query, document1, document2) -> float:
        ranking_data = _preference_vectors(context=context ,
                                           document1=document1 ,
                                           document2=document2 ,
                                           query=query ,
                                           embedding_style=s.SBERT ,
                                           comparison_method= s.MAX ,
                                           identifier=s.IDENT_AUS_SINGLE_SENTENCE

                                           )
        doc1_similarity = ranking_data[s.SOCKET_DOCUMENT1]
        doc2_similarity = ranking_data[s.SOCKET_DOCUMENT2]
        return strictly_greater(
            doc1_similarity,
            doc2_similarity
        )

class QArgSim_max_exact_sbert_full_document(Axiom):
    name = "QArgSim_max_exact_sbert_full_document"

    def preference(self, context, query, document1, document2) -> float:
        ranking_data = _preference_vectors(context=context,
                                           document1=document1,
                                           document2=document2,
                                           query=query,
                                           embedding_style=s.SBERT,
                                           comparison_method= s.MAX,
                                           document_sentenice=False,
                                           identifier=s.IDENT_AUS_FULL_DOCUMENT,
                                           task_info={s.TASK_INFO_TARGER_OWN_SENTENCIZER : True}

                                           )
        doc1_similarity = ranking_data[s.SOCKET_DOCUMENT1]
        doc2_similarity = ranking_data[s.SOCKET_DOCUMENT2]
        return strictly_greater(
            doc1_similarity,
            doc2_similarity
        )

class QArgSim_max_sbert(Axiom):
    name = "QArgSim_max_approxequal_sbert"

    def preference(self, context, query, document1, document2) -> float:
        ranking_data = _preference_vectors(context=context ,
                                           document1=document1 ,
                                           document2=document2 ,
                                           query=query ,
                                           embedding_style=s.SBERT ,
                                           comparison_method= s.MAX ,
                                           identifier=s.IDENT_AUS_SINGLE_SENTENCE

                                           )

        doc1_similarity = ranking_data[s.SOCKET_DOCUMENT1]
        doc2_similarity = ranking_data[s.SOCKET_DOCUMENT2]
        if approximately_equal(doc1_similarity,doc2_similarity):
            return 0
        return strictly_greater(
            doc1_similarity,
            doc2_similarity
        )

    
class QArgSim_max_sbert_full_document(Axiom):
    name = "QArgSim_max_approxequal_sbert_full_document"

    def preference(self, context, query, document1, document2) -> float:
        ranking_data = _preference_vectors(context=context,
                                           document1=document1,
                                           document2=document2,
                                           query=query,
                                           embedding_style=s.SBERT,
                                           comparison_method= s.MAX,
                                           document_sentenice=False,
                                           identifier=s.IDENT_AUS_FULL_DOCUMENT,
                                           task_info={s.TASK_INFO_TARGER_OWN_SENTENCIZER : True}
                                           )

        doc1_similarity = ranking_data[s.SOCKET_DOCUMENT1]
        doc2_similarity = ranking_data[s.SOCKET_DOCUMENT2]
        if approximately_equal(doc1_similarity,doc2_similarity):
            return 0
        return strictly_greater(
            doc1_similarity,
            doc2_similarity
        )
