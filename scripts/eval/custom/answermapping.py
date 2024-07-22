from lm_eval.api.filter import Filter
import pdb
from lm_eval.api.registry import register_filter

@register_filter("answermapping")
class AnswerMappingFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst, doc):
            mapped_resps = []
            for resp in inst:
                # if it is already a valid response, keep it
                if resp in doc['choices']['label']:
                    mapped_resps.append(resp)
                elif resp in doc['choices']['text']:
                    index = doc['choices']['text'].index(resp)
                    mapped_resps.append(doc['choices']['label'][index])
                else:
                    mapped_resps.append("[invalid]") 
            return mapped_resps

        filtered_resps = []
        for i, resp in enumerate(resps):
            filtered_resps.append(filter_set(resp, docs[i]))

        flat_filtered_resps = [item for sublist in filtered_resps for item in sublist] if any(isinstance(i, list) for i in filtered_resps) else filtered_resps

        return flat_filtered_resps