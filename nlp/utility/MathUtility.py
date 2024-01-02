from nlp.utility.Predefine import Predefine
from nlp.dictionary.CoreBiGramTableDictionary import CoreBiGramTableDictionary

class MathUtility():
    
    coreBiGramTableDictionary = CoreBiGramTableDictionary()
    
    @staticmethod
    def calculateWeight(_from, _to) -> float:
        fFrom = _from.getAttribute().totalFrequency
        fBigram = MathUtility.coreBiGramTableDictionary.getBiFrequency(_from.wordID, _to.wordID)
        fTo   = _to.getAttribute().totalFrequency

        return - math.log(Predefine._lambda * (Predefine.myu * fBigram / (fFrom + 1) + 1 - Predefine.myu) + (1 - Predefine._lambda) * fTo / Predefine.TOTAL_FREQUENCY)
