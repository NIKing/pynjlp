from nlp.seg.common.Edge import Edge

"""
记录了起点的边
"""
class EdgeFrom(Edge):

    _from = 0

    def __init__(self, _from, weight, name):
        super().__init__(weight, name)
        self._from = _from

    def toString(self):
        return "EdgeFrom{" \
                f"_from = {self._from}" \
                f",weight = {self.weight}" \
                f",name = {self.name}" \
                "}"
