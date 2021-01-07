def raise_spacy_component_does_not_exist(component_name: str):
    raise ValueError(
        f"The {component_name} component must exist in the provided spaCy model but does not. Check the components in model `spacy_model` by running `spacy_model.pipe_names()`"
    )
