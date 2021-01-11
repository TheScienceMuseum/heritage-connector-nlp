import spacy
from spacy.pipeline import EntityRuler
from spacy.language import Language
import time
from typing import List
from hc_nlp import constants, logging

logger = logging.get_logger(__name__)


@Language.factory(
    "thesaurus_matcher",
    default_config={"overwrite_ents": False, "case_sensitive": False},
)
def thesaurus_matcher(
    nlp, name, thesaurus_path: str, case_sensitive: bool, overwrite_ents: bool
):
    """
    Factory function for a ThesaurusMatcher.

    The ThesaurusMatcher lets you add spans to `Doc.ents` using exact phrase
    matches from an imported phrasebook (Thesaurus). It can be combined with 
    the statistical `EntityRecognizer` to boost accuracy, or used on its own 
    to implement a purely rule-based entity recognition system. After 
    initialization, the component is typically added to the pipeline using 
    `nlp.add_pipe`.
    """

    logger.info(f"Loading thesaurus from {thesaurus_path}")

    start = time.time()

    # set config for new entityruler object
    if case_sensitive:
        ruler = EntityRuler(nlp, overwrite_ents=overwrite_ents).from_disk(
            thesaurus_path
        )
    else:
        ruler = EntityRuler(
            nlp, overwrite_ents=overwrite_ents, phrase_matcher_attr="LOWER"
        ).from_disk(thesaurus_path)

    end = time.time()
    logger.info(f"{len(ruler)} term thesaurus imported in {int(end-start)}s")

    return ruler


@Language.factory("EntityFilter")
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
        nlp,
        name: str,
        max_token_length: int = 1,
        remove_all_lower: bool = True,
        remove_all_upper: bool = True,
        ent_labels_ignore: List[str] = [],
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
            ent_labels_ignore (List[str], optional): Entities with labels to ignore 
                when making the corrections.
        """
        self.nlp = nlp
        self.max_token_length = max_token_length
        self.remove_all_lower = remove_all_lower
        self.remove_all_upper = remove_all_upper
        self.ent_labels_ignore = ent_labels_ignore

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
            if ent.label_.upper() in self.ent_labels_ignore:
                likely_entity.append(True)
            else:
                likely_entity.append(
                    any([not self._is_unlikely_entity(tok) for tok in ent])
                )

        doc.ents = [ent for idx, ent in enumerate(doc.ents) if likely_entity[idx]]

        return doc


@Language.factory("pattern_matcher")
def pattern_matcher(nlp, name: str, patterns: List[dict]):
    """
    Create an EntityRuler object loaded with a list of patterns.

    Args:
        nlp : Spacy model
        patterns (List[dict]): for the EntityRuler. See https://spacy.io/usage/rule-based-matching#entityruler 

    Returns:
        Spacy EntityRuler component
    """
    ruler = EntityRuler(nlp)
    ruler.add_patterns(patterns)

    return ruler


@Language.factory("date_matcher")
def date_matcher(nlp, name) -> spacy.tokens.Doc:
    """
    A component designed to match dates through pattern matching and rules for detecting centuries.
    """
    return DateMatcher(nlp)


class DateMatcher:
    def __init__(self, nlp):
        self.ruler = EntityRuler(nlp)
        self.ruler.add_patterns(constants.DATE_PATTERNS)

    def _add_centuries_to_doc(self, doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        """
        Adds dates with the format "... nth century" to `Doc.ents`. Does this by finding the word
        'century' or 'centuries', checking that the previous word is an ordinal, and then returning all 
        the immediate children of the token 'century'. It then checks for occurrences of "nth (and/or/to) 
        mth centuries", as well as "AD" or "BC" after the word century/centuries.
        Args:
            doc (spacy.tokens.Doc)
        Returns:
            spacy.tokens.Doc
        """
        for idx, token in enumerate(doc):
            if token.lower_ in ["century", "centuries"]:
                if (doc[idx - 1].lower_ in constants.ORDINALS) or all(
                    [
                        string in constants.ORDINALS
                        for string in doc[idx - 1].lower_.split("-")
                    ]
                ):
                    try:
                        first_child = next(token.children)
                    except Exception:  # noqa: E722
                        # if the token has no children, use the ordinal token as first_child
                        first_child = doc[idx - 1]

                    # allow "nth (and|to|or) mth" century
                    if (doc[first_child.i - 1].lower_ in ["and", "to", "or"]) and (
                        doc[first_child.i - 2].lower_ in constants.ORDINALS
                    ):
                        # go back to the first child of "nth"
                        start = next(doc[first_child.i - 2].children).i

                        # if the child is after the 'nth' token, use the token instead of its child
                        if start > doc[first_child.i - 2].i:
                            start = doc[first_child.i - 2].i
                    else:
                        start = first_child.i

                    # print([c for c in doc[first_child.i - 2].children])

                    end = idx + 1

                    # add on 'AD' or 'BC' if present
                    if idx != len(doc) - 1:
                        if doc[idx + 1].text.upper() in ["AD", "BC"]:
                            end += 1

                    date_entity = spacy.tokens.Span(doc, start, end, label="DATE")
                    try:
                        doc.ents = list(doc.ents) + [date_entity]
                    except Exception:
                        # TODO: check for overlap instead of just failing
                        # TODO: handle the specific spaCy error
                        logger.warn(
                            f"Failed to add DATE entity {date_entity.text} in pos {(start, end)} to text {doc.text}"
                        )
        return doc

    def __call__(self, doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        """
        Detects centuries then patterns from `constants.DATE_PATTERNS`.
        Args:
            doc (spacy.tokens.Doc)
        Returns:
            spacy.tokens.Doc
        """
        doc = self._add_centuries_to_doc(doc)
        doc = self.ruler(doc)

        return doc


@Language.factory("MapEntityTypes")
class MapEntityTypes:
    def __init__(
        self, nlp, name: str, mapping: dict = constants.SPACY_TO_HC_ENTITY_MAPPING,
    ):

        self.mapping = mapping
        self.nlp = nlp

    def __call__(self, doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        """
        Replace entities in `Doc.ents` with new entities based on the mapping specified when initialising
        the class instance.

        Args:
            doc (spacy.tokens.Doc

        Returns:
            spacy.tokens.Doc
        """
        new_ents = []
        for ent in list(doc.ents):
            new_ent = spacy.tokens.Span(
                doc, ent.start, ent.end, label=self.mapping.get(ent.label_, ent.label_)
            )
            new_ents.append(new_ent)

        doc.ents = new_ents

        return doc
