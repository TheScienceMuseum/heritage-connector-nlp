### PATTERNS FOR FINDING ENTITIES OF SPECIFIC TYPES

DATE_PATTERNS = [
    {"label": "DATE", "pattern": [{"SHAPE": "dddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},  # 1984 - 1990 | 1984-1990
    {"label": "DATE", "pattern": [{"ORTH": "c."}, {"SHAPE": "dddd"}]},  # c. 1200
    {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"c\.\d{3,4}"}}]},  # c.1200
    {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"c\.\d{3,4}"}}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},  # c.1200 - 1220 | c.1200-1220
    {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/(\d{4}|\d{2})"}}]},  # 03/12/2000
    {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}\.\d{1,2}\.(\d{4}|\d{2})"}}]},  # 03.12.2000
    {"label": "DATE", "pattern": [{"SHAPE": "dd"}, {"ORTH": "-"}, {"SHAPE": "dd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},  # 03-12-2000
    {"label": "DATE", "pattern": [{"SHAPE": "d"}, {"ORTH": "-"}, {"SHAPE": "dd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},  # 3-12-2000
    {"label": "DATE", "pattern": [{"SHAPE": "dd"}, {"ORTH": "-"}, {"SHAPE": "d"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},  # 03-1-2000
    {"label": "DATE", "pattern": [{"SHAPE": "d"}, {"ORTH": "-"}, {"SHAPE": "d"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},  # 3-1-2000
    {"label": "DATE", "pattern": [{"SHAPE": "dddd"}, {"ORTH": "to"}, {"SHAPE": "dddd"}]},  # 1805 to 1860
]

COLLECTION_NAME_PATTERNS = [
    # TODO: use 'POS': 'PROPN' here instead of IS_TITLE: True for better detection of proper nouns
    {"label": "ORG", "pattern": [{'IS_TITLE': True, 'OP': '+'}, {'LOWER': 'collection'}]},  # Sforza collection
    {"label": "ORG", "pattern": [{'IS_TITLE': True, 'OP': '+'}, {'LOWER': 'archive'}]},  # Charles Urban archive
]

### USEFUL CONSTANTS

ORDINALS = [
    "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
    "11th", "12th", "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th", "21st",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
    "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
    "seventeenth", "eighteenth", "nineteenth", "twentieth", "twenty-first"
]

ROYAL_TITLES = [
    "king", "queen", "prince", "princess", "emperor", "empress"
]

### SPACY TO HERITAGECONNECTOR ENTITY MAPPING
### This allows us to map multiple Spacy NER classes to one class, in the case that we don't need the 
### detail of specific classes. Also includes identity mappings. `set(SPACY_TO_HC_ENTITY_MAPPING.values())`
### is the complete set of entities are using for the Heritage Connector.
SPACY_TO_HC_ENTITY_MAPPING = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "NORP": "NORP",
    "GPE": "LOC",
    "LOC": "LOC",
    "PRODUCT": "OBJECT",
    "WORK_OF_ART": "OBJECT",
    "EVENT": "EVENT",
    "DATE": "DATE",
}
