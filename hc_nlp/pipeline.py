import spacy
from spacy.pipeline import EntityRuler
from spacy.language import Language
import time
import copy
from typing import Sequence
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
    other_pipes = [p for p in nlp.pipe_names if p != "tagger"]

    start = time.time()

    # set config for new entityruler object
    if case_sensitive:
        with nlp.select_pipes(disable=other_pipes):
            ruler = EntityRuler(nlp, overwrite_ents=overwrite_ents).from_disk(
                thesaurus_path
            )
    else:
        with nlp.select_pipes(disable=other_pipes):
            ruler = EntityRuler(
                nlp, overwrite_ents=overwrite_ents, phrase_matcher_attr="LOWER"
            ).from_disk(thesaurus_path)

    end = time.time()
    logger.info(f"{len(ruler)} term thesaurus imported in {int(end-start)}s")

    return ruler


@Language.factory("entity_filter")
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
        remove_all_upper: bool = False,
        ent_labels_ignore: Sequence[str] = [],
    ):
        """
        Initialise the EntityFilter.

        Args:
            max_token_length (int, optional): Entities with tokens with length less
                than or equal to this will be removed from Doc.ents. Defaults to 1.
            remove_all_lower (bool, optional): Entities with one or more lowercase
                token are removed. Defaults to True.
            remove_all_upper (bool, optional): Entities with one or more uppercase
                token are removed. Defaults to False.
            ent_labels_ignore (Sequence[str], optional): Entities with labels to ignore
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

    def _remove_the_year_from_date_entities(
        self, doc: spacy.tokens.Doc
    ) -> spacy.tokens.Doc:
        """
        Removes phrase 'the year' from the start of any DATE entities.
        """

        newdoc = copy.copy(doc)

        for ent in newdoc.ents:
            if (ent.label_ == "DATE") and ent.text.lower().startswith("the year"):
                if ent.text.lower() == "the year":
                    # remove entire entity
                    newdoc.ents = [e for e in newdoc.ents if e != ent]
                else:
                    new_ent = spacy.tokens.Span(
                        newdoc, ent.start + 2, ent.end, label="DATE"
                    )

                    newdoc.ents = [new_ent if e == ent else e for e in newdoc.ents]

        return newdoc

    def _remove_n_years_from_date_entities(
        self, doc: spacy.tokens.Doc
    ) -> spacy.tokens.Doc:
        """
        Removes any DATE entities with the format 'n years'
        """

        newdoc = copy.copy(doc)

        for ent in newdoc.ents:
            if (
                (ent.label_ == "DATE")
                and ("years" in ent.text.lower())
                and ent[0].like_num
            ):
                newdoc.ents = [e for e in newdoc.ents if e != ent]

        return newdoc

    def _add_royal_title_to_person_entities(
        self, doc: spacy.tokens.Doc
    ) -> spacy.tokens.Doc:
        newdoc = copy.copy(doc)

        for ent in newdoc.ents:
            if ent.label_ == "PERSON":
                if (newdoc[ent.start - 1].text.lower() in constants.ROYAL_TITLES) and (
                    not newdoc[ent.start - 1].ent_type_
                ):
                    new_ent = spacy.tokens.Span(
                        newdoc, ent.start - 1, ent.end, label="PERSON"
                    )

                    newdoc.ents = [new_ent if e == ent else e for e in newdoc.ents]

        return newdoc

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

        doc = self._remove_the_year_from_date_entities(doc)
        doc = self._add_royal_title_to_person_entities(doc)
        doc = self._remove_n_years_from_date_entities(doc)

        return doc


def pattern_matcher(nlp, name: str, patterns: Sequence[dict]):
    """
    Create an EntityRuler object loaded with a list of patterns.

    Args:
        nlp : Spacy model
        patterns (Sequence[dict]): for the EntityRuler. See https://spacy.io/usage/rule-based-matching#entityruler

    Returns:
        Spacy EntityRuler component
    """
    ruler = EntityRuler(nlp)
    ruler.add_patterns(patterns)

    return ruler


@Language.factory("pattern_matcher")
class PatternMatcher:
    """
    An EntityRuler object initiated with a pattern. Used for built-in `hc_nlp`
    matchers.
    """

    def __init__(self, nlp, name: str, patterns: Sequence[dict]):
        """
        Initialise the PatternMatcher.

        Args:
            nlp : Spacy model
            patterns (Sequence[dict]): for the EntityRuler. See https://spacy.io/usage/rule-based-matching#entityruler
        """
        self.ruler = EntityRuler(nlp)
        self.ruler.add_patterns(patterns)

    def __call__(self, doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        """
        Inherits from EntityRuler behaviour.
        """
        return self.ruler(doc)


@Language.factory("date_matcher")
class DateMatcher(PatternMatcher):
    def __init__(self, nlp, name):
        # TODO: inherit from pattern_matcher
        super().__init__(nlp, name, constants.DATE_PATTERNS)
        # self.ruler = EntityRuler(nlp)
        # self.ruler.add_patterns(constants.DATE_PATTERNS)

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
                        try:
                            # go back to the first child of "nth"
                            start = next(doc[first_child.i - 2].children).i

                            # if the child is after the 'nth' token, use the token instead of its child
                            if start > doc[first_child.i - 2].i:
                                start = doc[first_child.i - 2].i
                        except Exception:
                            # if couldn't find children of 'nth', then just take 'nth' as start
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


@Language.factory("map_entity_types")
class MapEntityTypes:
    def __init__(
        self,
        nlp,
        name: str,
        mapping: dict = constants.SPACY_TO_HC_ENTITY_MAPPING,
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


@Language.factory("entity_joiner")
class EntityJoiner:
    """
    A pipeline element which operates on doc objects with entities already annotated. It:
    - joins consecutive entities which have the same label;
    - joins pairs of location entities (by default those with label LOC) which are separated by only a comma.
    - for consecutive PERSON entities separated by an 'and' (e.g. 'Katharine and Charles Parsons'), sets the
    span attribute `ent._.alt_ent_text` to the full name of the first person ('Katharine Parsons',
    using the same example). Useful for entity linking.
    """

    def __init__(self, nlp, name):
        self.nlp = nlp

    def _detect_joined_person_entities(self, doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        """Detect two people in a row separated by an 'and', where the first person is only referred to by their
        first name. Set the attribute `ent._.alt_ent_text` for the first person to their first name, plus
        the surname of the next mentioned person.

        Args:
            doc (spacy.tokens.Doc)

        Returns:
            spacy.tokens.Doc: amended doc
        """
        # set custom span attributes
        if not spacy.tokens.Span.has_extension("alt_ent_text"):
            spacy.tokens.Span.set_extension("alt_ent_text", default=None)

        idx = 0
        new_ents = []

        while idx < len(doc.ents):
            if idx + 1 == len(doc.ents):
                new_ents.append(doc.ents[idx])
                idx += 1
                continue

            if doc.ents[idx].end >= len(doc):
                idx += 1
                continue

            curr_ent = doc.ents[idx]
            next_token = doc[curr_ent.end]
            next_ent = doc.ents[idx + 1]

            # two consecutive entities are labelled PERSON; separated only by 'and' or '&'; don't share the same last token (i.e. surname)
            if (
                (curr_ent.label_ == next_ent.label_ == "PERSON")
                and (next_token.text.lower() in {"and", "&"})
                and (curr_ent.end + 1 == next_ent.start)
                and (curr_ent[-1].text.lower() != next_ent[-1].text.lower())
            ):
                # assume lastname is all tokens but the first
                lastname = next_ent[1:].text
                curr_ent._.alt_ent_text = curr_ent.text + " " + lastname

            new_ents.append(curr_ent)
            idx += 1

        newdoc = copy.copy(doc)
        newdoc.ents = new_ents

        return newdoc

    def _join_consecutive_ents_with_same_label(
        self, doc: spacy.tokens.Doc, exclude_types: Sequence[str] = []
    ) -> spacy.tokens.Doc:
        """Join entities which occupy consecutive tokens and have the same label.

        Args:
            doc (spacy.tokens.Doc)
            exclude_types (Sequence[str]): entity labels for which consecutive tokens should not be joined.

        Returns:
            spacy.tokens.Doc: amended doc
        """
        idx = 0
        new_ents = []

        while idx < len(doc.ents):
            # add last entity to new_ents as not included in above while loop
            if idx + 1 == len(doc.ents):
                new_ents.append(doc.ents[idx])
                idx += 1
                continue

            if doc.ents[idx].end >= len(doc):
                idx += 1
                continue

            curr_ent = doc.ents[idx]
            next_token = doc[curr_ent.end]

            if curr_ent.label_ == next_token.ent_type_:
                # search for continuation of the same label for future tokens, starting with the one after
                # the next token. For each token with a matching label found, increment the offset value by one.
                # This is then used to set the end of joined_ent and increment the value of idx.
                next_token_offset = 0
                while ((curr_ent.end + 1 + next_token_offset) < len(doc)) and (
                    curr_ent.label_
                    == doc[curr_ent.end + 1 + next_token_offset].ent_type_
                ):
                    next_token_offset += 1

                joined_ent_end = curr_ent.end + 1 + next_token_offset
                joined_ent = spacy.tokens.Span(
                    doc,
                    curr_ent.start,
                    joined_ent_end,
                    curr_ent.label_,
                )
                new_ents.append(joined_ent)

                # find and go to next entity after the observed span is finished
                ent_idxs_after_joined_ent = [
                    i for (i, e) in enumerate(doc.ents) if e.start > joined_ent_end
                ]

                if len(ent_idxs_after_joined_ent) > 0:
                    idx = min(ent_idxs_after_joined_ent)
                else:
                    break

            else:
                new_ents.append(curr_ent)
                idx += 1

        newdoc = copy.copy(doc)
        newdoc.ents = new_ents

        return newdoc

    def _join_comma_separated_locs(
        self, doc: spacy.tokens.Doc, loc_ent_labels: Sequence[str] = ["LOC"]
    ) -> spacy.tokens.Doc:
        """
        Join pairs of consecutive LOC entities which are separated by only a comma, e.g. "[Brighton], [UK]" -> "[Brighton, UK]".
        Ignores spaces around the comma.

        Args:
            doc (spacy.tokens.Doc):
            loc_ent_labels (Sequence[str], optional): entity label names for location entities. Defaults to ["LOC"].

        Returns:
            spacy.tokens.Doc:
        """
        idx = 0
        new_ents = []

        while idx < len(doc.ents):
            # add last entity to new_ents as not included in above while loop
            if idx + 1 == len(doc.ents):
                new_ents.append(doc.ents[idx])
                idx += 1
                continue

            # stop at token before last token (as operates on minimum 3 consecutive tokens)
            if doc.ents[idx].end + 1 >= len(doc):
                idx += 1
                continue

            curr_ent = doc.ents[idx]
            next_ent = doc.ents[idx + 1]
            next_token = doc[curr_ent.end]
            token_after_next_token = doc[curr_ent.end + 1]

            # allow for extra spaces either side of the comma
            if (
                (curr_ent.label_ in loc_ent_labels)
                and (next_token.text.strip() == ",")
                and (next_ent.label_ in loc_ent_labels)
                and (token_after_next_token.ent_type_ in loc_ent_labels)
            ):

                joined_loc_ent = spacy.tokens.Span(
                    doc, curr_ent.start, next_ent.end, curr_ent.label_
                )
                new_ents.append(joined_loc_ent)

                idx += 2

            else:
                new_ents.append(curr_ent)
                idx += 1

        newdoc = copy.copy(doc)
        newdoc.ents = new_ents

        return newdoc

    def __call__(self, doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        newdoc = copy.copy(doc)
        newdoc = self._join_consecutive_ents_with_same_label(newdoc)
        newdoc = self._join_comma_separated_locs(newdoc)
        newdoc = self._detect_joined_person_entities(newdoc)

        return newdoc


@Language.factory("duplicate_entity_detector")
class DuplicateEntityDetector:
    """
    A pipeline element to detect multiple mentions of the same real-world entity (of certain types).
    It operates on documents which already have annotated entities. DuplicateEntityDetector sets two
    custom attributes for spans within the Doc object:
    - `span._.entity_co_occurrence`: entities in a document with the same value of this attribute are predicted
    to refer to the same real-world entity. Defaults to None (span is not part of a co-occurrence.)
    - `span._.entity_duplicate`: set to True if a labelled entity is predicted to be a duplicate of one before it
    in the document. Duplicates are captured through a second, shorter mention of the entity. Defaults to False.

    E.g. in a document with 'Joseph Henry' (PERSON) followed by 'Henry' (PERSON) or 'Joseph' (PERSON) later on in
    the passage, the `span._.entity_co_occurrence` attribute will be set to the same string value for both entities
    and the `span._.entity_duplicate` attribute will be set to False for the first mention and True for the second.
    """

    def __init__(self, nlp, name, types_ignore: Sequence[str] = []):
        """Create an instance of DuplicateEntityDetector.

        Args:
            nlp, name
            types_ignore (Sequence [str], optional): Entity types to ignore from ["PERSON", "ORG", "LOC"].
        """

        self.types_ignore = {"PERSON", "ORG", "LOC"}.intersection(set(types_ignore))

        # set custom span attributes
        if not spacy.tokens.Span.has_extension("entity_co_occurrence"):
            spacy.tokens.Span.set_extension("entity_co_occurrence", default=None)

        if not spacy.tokens.Span.has_extension("entity_duplicate"):
            spacy.tokens.Span.set_extension("entity_duplicate", default=False)

    def _detect_duplicate_person_mentions(
        self, doc: spacy.tokens.Doc
    ) -> spacy.tokens.Doc:
        """
        Detect duplicate person mentions of the pattern "Firstname Lastname" then "Lastname".
        Marks both with a `entity_co_occurrence` value "firstname_lastname" and all but the first
        mention with a `entity_duplicate` value of True.

        Args:
            doc (spacy.tokens.Doc)

        Returns:
            spacy.tokens.Doc
        """
        newdoc = copy.copy(doc)
        co_occurrence_string = (
            lambda firstname, lastname: f"{firstname.lower()}_{lastname.lower()}"
        )

        for idx, ent in enumerate(newdoc.ents):
            found_entity_co_occurrence = False

            # Check for alternative entity text (i.e. firstname + lastname) which will have
            # been inserted if EntityJoiner was applied before DuplicateEntityDetector in the
            # pipeline.
            if (
                spacy.tokens.Span.has_extension("alt_ent_text")
                and ent._.alt_ent_text is not None
            ):
                len_ent = len(ent._.alt_ent_text.split(" "))
            else:
                len_ent = len(ent)

            if (ent.label_ == "PERSON") and (len_ent > 1):
                # Assumes first name is only one word and all other words make up the last name.
                if (
                    spacy.tokens.Span.has_extension("alt_ent_text")
                    and ent._.alt_ent_text is not None
                ):
                    ent_text_split = ent._.alt_ent_text.split(" ")
                    firstname = ent_text_split[0]
                    lastname = " ".join(ent_text_split[1:])
                else:
                    firstname = ent[0].text
                    lastname = ent[1:].text

                # find other entities with text equal to firstname or lastname. Only look at entities in the
                # doc that occur after the current entity.
                for e in newdoc.ents[idx + 1 :]:
                    if (e != ent) and (
                        (e.text.lower() == lastname.lower())
                        or (e.text.lower() == firstname.lower())
                    ):
                        found_entity_co_occurrence = True

                        e = spacy.tokens.Span(
                            newdoc, start=e.start, end=e.end, label="PERSON"
                        )
                        e._.entity_co_occurrence = co_occurrence_string(
                            firstname, lastname
                        )
                        e._.entity_duplicate = True

                if found_entity_co_occurrence:
                    ent._.entity_co_occurrence = co_occurrence_string(
                        firstname, lastname
                    )

        return newdoc

    def _detect_duplicate_org_mentions(self, doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        """Detect duplicate organisation mentions where one ORG entity in the doc is of the form "<company name> <legal suffix>",
        and there are other ORG entities of the form "<company name>".

        Treats the name with the suffix as the main one and marks all others as duplicate, setting the `entity_co_occurrence`
        attribute to "company_name_legal_suffix" (underscore-joined and lowercased).

        Args:
            doc (spacy.tokens.Doc)

        Returns:
            spacy.tokens.Doc
        """

        newdoc = copy.copy(doc)
        co_occurrence_string = lambda ent: "_".join(
            [i.lower() for i in ent.text.split(" ")]
        )

        for idx, ent in enumerate(newdoc.ents):
            found_co_occurrence = False

            if (ent.label_ == "ORG") and (len(ent) > 1):
                if ent[-1].text.lower() in [
                    s.lower() for s in constants.ORG_LEGAL_SUFFIXES
                ]:
                    org_without_suffix = ent[0:-1]

                    # Find other entities matching just org without suffix.
                    # Enforce that these are already predicted to be ORGs to avoid overwriting places and people
                    # which might have organisations named after them.
                    for e in newdoc.ents:
                        if (
                            (e != ent)
                            and (e.label_ == "ORG")
                            and (e.text.lower() == org_without_suffix.text.lower())
                        ):
                            found_co_occurrence = True
                            e._.entity_co_occurrence = co_occurrence_string(ent)
                            e._.entity_duplicate = True

                    if found_co_occurrence:
                        ent._.entity_co_occurrence = co_occurrence_string(ent)

        return newdoc

    def _detect_duplicate_loc_mentions(self, doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        """Detect duplicate location (LOC) mentions where one LOC entity in the doc is of the form "<place>, <surrounding place>",
        and there are other LOC entities of the form "<place>".

        Treats the longer entity mention as the main one and marks all others as duplicate, setting the `entity_co_occurrence`
        attribute to the underscore-joined and lowercased version of the longest entity mention.

        Args:
            doc (spacy.tokens.Doc)

        Returns:
            spacy.tokens.Doc
        """

        newdoc = copy.copy(doc)
        co_occurrence_string = lambda ent: "_".join(
            [i.lower() for i in ent.text.split(" ")]
        )

        for idx, ent in enumerate(newdoc.ents):
            found_co_occurrence = False

            if (ent.label_ == "LOC") and (len(ent) > 1) and ("," in ent.text):
                loc_first_part = ent.text.split(",")[0]

                for e in newdoc.ents:
                    if (
                        (e != ent)
                        and (e.label_ == "LOC")
                        and (e.text.lower() == loc_first_part.lower())
                    ):
                        found_co_occurrence = True
                        e._.entity_co_occurrence = co_occurrence_string(ent)
                        e._.entity_duplicate = True

                if found_co_occurrence:
                    ent._.entity_co_occurrence = co_occurrence_string(ent)

        return newdoc

    def __call__(self, doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        newdoc = copy.copy(doc)

        if "PERSON" not in self.types_ignore:
            newdoc = self._detect_duplicate_person_mentions(newdoc)

        if "ORG" not in self.types_ignore:
            newdoc = self._detect_duplicate_org_mentions(newdoc)

        if "LOC" not in self.types_ignore:
            newdoc = self._detect_duplicate_loc_mentions(newdoc)

        return newdoc
