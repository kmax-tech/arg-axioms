from ir_axioms.modules.pivot import MiddlePivotSelection
from ir_axioms.backend.pyterrier.transformers import KwikSortReranker, RandomPivotSelection
import settings as s


def kwiksort(retr_sys, axioms,nbr):
    result_list = []
    for axiom in axioms:

        # detailed version of the Kwiksort reranker
        kwik_rerank = KwikSortReranker(
        axiom=axiom,
        index=s.dataset_index_dir,
        pivot_selection=MiddlePivotSelection(),
        cache_dir=s.cache_dir
        )

        rs = (retr_sys % nbr >>  kwik_rerank) ^ retr_sys
        result = retr_sys % nbr >>  kwik_rerank ^ retr_sys
        result2 = ((retr_sys % nbr) >>  kwik_rerank) ^ retr_sys
        result3 = retr_sys >> (kwik_rerank ^ retr_sys)

        result_list.append(result)
    return result_list


def kwiksort_orig(retr_sys, axioms,nbr):
    return [retr_sys % nbr >> KwikSortReranker(
        axiom=axiom,
        index=s.dataset_index_dir,
        pivot_selection=MiddlePivotSelection(),
        cache_dir=s.cache_dir
    ) ^ retr_sys
    for axiom in axioms]
