from nlp.seg.common.Edge import Edge

"""
记录了起点的边
"""
class EdgeFrom(Edge):

    _from = 0
    _to = 0

    def __init__(self, _from, weight, name, _to):
        super().__init__(weight, name)
        self._from = _from
        self._to = _to

    def toString(self):
        return "EdgeFrom{" \
                f"_from: {self._from} " \
                f"weight: {self.weight} " \
                f"name: {self.name} " \
                f"_to: {self._to}" \
                "}"
