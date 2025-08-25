__all__ = ["PhonemeTransformJob"]

import xml.etree.ElementTree as ET

from sisyphus import *

from i6_core.util import uopen

Path = setup_path(__package__)


class PhonemeTransformJob(Job):
    """
    Transforms phonemes from monophone to diphone
    """

    def __init__(self, xml_lexicon: Path):
        """
        :param xml_lexicon: xml file to be processed
        """
        self.in_lexicon = xml_lexicon
        self.out_lexicon = self.output_path("lexicon.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        file = uopen(self.in_lexicon.get_path())
        tree = ET.parse(file)
        root = tree.getroot()

        # Extract phonemes
        phonemes = [phoneme.find('symbol').text for phoneme in root.find('phoneme-inventory').findall('phoneme')]

        # Generate diphones
        diphones = generate_diphones(phonemes)
        
        # Clear existing phoneme-inventory
        phoneme_inventory = root.find('phoneme-inventory')
        phoneme_inventory.clear()

        # Add diphones as new phonemes in phoneme-inventory
        for diphone in diphones:
            phoneme_element = ET.Element('phoneme')
            symbol_element = ET.SubElement(phoneme_element, 'symbol')
            symbol_element.text = diphone
            phoneme_inventory.append(phoneme_element)

        # Replace phonemes in lemmas with diphones for all pronunciations
        for lemma in root.findall('.//lemma'):
            for phon_element in lemma.findall('phon'):
                phon_text = phon_element.text
                transformed_phon = transform_phoneme(phon_text)
                phon_element.text = transformed_phon


        # Save the modified tree to a new file?
        tree.write(self.out_lexicon.get_path())

    
def transform_phoneme(phon_str: str) -> str:
    """
    Transforms the phoneme string according to diphone use.
    """
    phonemes = phon_str.split()
    transformed = ['#_' + phonemes[0]] + [phonemes[i] + '_' + phonemes[i + 1] for i in range(len(phonemes) - 1)]
    return ' '.join(transformed)
   
    
def generate_diphones(phonemes):
    diphones = []
    for p1 in phonemes:
        diphones.append('#_' + p1)
        for p2 in phonemes:
            diphones.append(p1 + '_' + p2)
    return diphones
