import spacy
from spacy.pipeline import EntityRuler
import time


class ThesaurusMatcher:
    """
    The ThesaurusMatcher lets you add spans to `Doc.ents` using exact phrase
    matches from an imported phrasebook (Thesaurus). It can be combined with 
    the statistical `EntityRecognizer` to boost accuracy, or used on its own 
    to implement a purely rule-based entity recognition system. After 
    initialization, the component is typically added to the pipeline using 
    `nlp.add_pipe`.
    """

    def __init__(self, nlp, thesaurus_path: str, case_sensitive: bool):
        """
        Initialise the ThesaurusMatcher. `thesaurus_path` must point to a .jsonl
        file with each line in the following format (with the `id` key optional):

        ```
        {
            "label": "<entity label>",
            "pattern": "<text to match>",
            "id": "<ID which the above text unambiguously refers to. Optional.>",
        }
        ```

        Args:
            nlp: spacy model
            thesaurus_path (str): path to the thesaurus
            case_sensitive (bool): [description]
        """
        if case_sensitive:
            self.ruler = EntityRuler(nlp)
        else:
            self.ruler = EntityRuler(nlp, phrase_matcher_attr="LOWER")

        self.nlp = nlp
        self.thesaurus_path = thesaurus_path
        self._add_thesaurus_to_ruler()

    def _add_thesaurus_to_ruler(self):
        """
        Load thesaurus from disk and add to self.ruler
        """
        print(f"Loading thesaurus from {self.thesaurus_path}")
        other_pipes = [p for p in self.nlp.pipe_names if p != "tagger"]

        start = time.time()
        with self.nlp.disable_pipes(*other_pipes):
            self.ruler.from_disk(self.thesaurus_path)

        end = time.time()
        print(f"{len(self.ruler)} term thesaurus imported in {int(end-start)}s")

    def __call__(self, doc):
        """
        Effectively inherits from EntityRuler behaviour.
        """
        return self.ruler(doc)


class EntityFilter:
    """
    The EntityFilter filters out any entities in `Doc.ents` that aren't likely to be
    entities based on their form. It uses the following rules to determine this, which
    can be switched off or changed when initialising the EntityFilter:
    - under `max_token_length` -> not entity
    - contains 3+ consecutive digits -> maybe entity (could be a date)
    - all lowercase or all UPPERCASE -> not entity

    These rules are applied so that if any of the tokens in an entity span don't look 
    like entities, the whole span will be removed from `Doc.ents`.
    """

    def __init__(
        self,
        max_token_length: int = 1,
        remove_all_lower: bool = True,
        remove_all_upper: bool = True,
    ):
        """
        Initialise the EntityFilter.

        Args:
            max_token_length (int, optional): Entities with tokens with length less
                than or equal to this will be removed from Doc.ents. Defaults to 1.
            remove_all_lower (bool, optional): Entities with one or more lowercase 
                token are removed. Defaults to True.
            remove_all_upper (bool, optional): Entities with one or more uppercase
                token are removed. Defaults to True.
        """
        self.max_token_length = max_token_length
        self.remove_all_lower = remove_all_lower
        self.remove_all_upper = remove_all_upper

    def _is_unlikely_entity(self, token: spacy.tokens.Token) -> bool:
        """
        Returns True if a token is likely not an entity, and False otherwise.
        """
        if "ddd" in token.shape_:
            # tokens with 3+ consecutive digits could be years, so possibly DATE entities
            return False

        elif len(token) <= self.max_token_length:
            return True

        # UPPERCASE and lowercase tokens assumed not to be entities
        elif token.is_lower and self.remove_all_lower:
            return True

        elif token.is_upper and self.remove_all_upper:
            return True

        return False

    def __call__(self, doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        """
        Filter out entities which contain one or more token that doesn't look like
        an entity.
        """
        likely_entity = []
        for ent in doc.ents:
            likely_entity.append(
                any([not self._is_unlikely_entity(tok) for tok in ent])
            )

        doc.ents = [ent for idx, ent in enumerate(doc.ents) if likely_entity[idx]]

        return doc
