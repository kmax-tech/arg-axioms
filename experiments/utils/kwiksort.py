from ir_axioms.modules.pivot import MiddlePivotSelection
from ir_axioms.backend.pyterrier.transformers import KwikSortReranker,RandomPivotSelection
import settings as s


def kwiksort(retr_sys, axioms,nbr):
    return [retr_sys % nbr >> KwikSortReranker(
        axiom=axiom,
        index=s.dataset_index_dir,
        pivot_selection=MiddlePivotSelection(),
        cache_dir=s.cache_dir
    ) ^ retr_sys
    for axiom in axioms]


