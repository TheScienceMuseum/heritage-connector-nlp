DATE_PATTERNS = [
    {"label": "DATE", "pattern": [{"SHAPE": "dddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},  # 1984 - 1990 | 1984-1990
    {"label": "DATE", "pattern": [{"ORTH": "c."}, {"SHAPE": "dddd"}]},  # c. 1200
    {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"c\.\d{3,4}"}}]},  # c.1200
    {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/(\d{4}|\d{2})"}}]},  # 03/12/2000
    {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}\.\d{1,2}\.(\d{4}|\d{2})"}}]},  # 03.12.2000
    {"label": "DATE", "pattern": [{"SHAPE": "dd"}, {"ORTH": "-"}, {"SHAPE": "dd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},  # 03-12-2000
    {"label": "DATE", "pattern": [{"SHAPE": "d"}, {"ORTH": "-"}, {"SHAPE": "dd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},  # 3-12-2000
    {"label": "DATE", "pattern": [{"SHAPE": "dd"}, {"ORTH": "-"}, {"SHAPE": "d"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},  # 03-1-2000
    {"label": "DATE", "pattern": [{"SHAPE": "d"}, {"ORTH": "-"}, {"SHAPE": "d"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},  # 3-1-2000
    {"label": "DATE", "pattern": [{"SHAPE": "dddd"}, {"ORTH": "to"}, {"SHAPE": "dddd"}]},  # 1805 to 1860
]

###

ORDINALS = [
    "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
    "11th", "12th", "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th", "21st",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
    "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
    "seventeenth", "eighteenth", "nineteenth", "twentieth", "twenty-first"
]
