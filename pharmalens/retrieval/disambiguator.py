"""Product/formulation intent disambiguation for semaglutide questions."""

PRODUCT_SIGNALS = {
    "Ozempic": ["ozempic", "type 2 diabetes", "diabetes", "t2dm", "glycaemic", "glycemic", "hba1c"],
    "Wegovy": ["wegovy", "weight", "obesity", "bmi", "weight loss", "weight management"],
    "Rybelsus": ["rybelsus", "oral semaglutide", "semaglutide tablet"],
}


def detect_product_intent(query: str) -> dict:
    q = query.lower()
    if any(term in q for term in ("paracetamol", "atorvastatin", "warfarin", "inr", "auc", "cmax", "interaction")):
        return {"status": "clear", "product": None, "formulation": None, "message": None}
    explicit_products = [product for product, signals in PRODUCT_SIGNALS.items() if signals[0] in q]
    if len(explicit_products) > 1:
        return {"status": "clear", "product": None, "formulation": None, "message": None}
    for product, signals in PRODUCT_SIGNALS.items():
        if signals[0] in q:
            return {"status": "clear", "product": product, "formulation": "OralTablet" if product == "Rybelsus" else "SubcutaneousSolution", "message": None}

    matched = [product for product, signals in PRODUCT_SIGNALS.items() if any(signal in q for signal in signals[1:])]
    if len(matched) == 1:
        product = matched[0]
        return {"status": "clear", "product": product, "formulation": "OralTablet" if product == "Rybelsus" else "SubcutaneousSolution", "message": None}

    if "semaglutide" in q and any(term in q for term in ("dose", "dosing", "posology", "indication", "contraindication", "administer")):
        return {
            "status": "ambiguous",
            "product": None,
            "formulation": None,
            "message": (
                "Your question could apply to multiple semaglutide products: Ozempic for type 2 diabetes, "
                "Wegovy for weight management, or Rybelsus/oral semaglutide tablets. Which product are you asking about?"
            ),
        }

    return {"status": "clear", "product": None, "formulation": None, "message": None}
