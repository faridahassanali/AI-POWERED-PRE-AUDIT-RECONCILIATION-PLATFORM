import pandas as pd

from engine.normalization import normalize_dataframe


def test_categorical_values_are_normalized():

    df = pd.DataFrame(
        {
            "wallet_status": [
                " OPENED ",
                "opened",
                "OPENED",
            ],
            "risk_level": [
                " high ",
                "HIGH",
                "High",
            ],
        }
    )

    result = normalize_dataframe(df)

    assert result["wallet_status"].tolist() == [
        "OPENED",
        "OPENED",
        "OPENED",
    ]

    assert result["risk_level"].tolist() == [
        "HIGH",
        "HIGH",
        "HIGH",
    ]


def test_boolean_values_are_normalized():

    df = pd.DataFrame(
        {
            "screening_evidence_present": [
                "true",
                "TRUE",
                "yes",
                "1",
                "false",
                "FALSE",
                "no",
                "0",
            ]
        }
    )

    result = normalize_dataframe(df)

    assert result[
        "screening_evidence_present"
    ].tolist() == [
        "True",
        "True",
        "True",
        "True",
        "False",
        "False",
        "False",
        "False",
    ]


def test_whitespace_is_trimmed():

    df = pd.DataFrame(
        {
            "customer_id": [
                " CUST100001 ",
                "CUST100002  ",
                "  CUST100003",
            ]
        }
    )

    result = normalize_dataframe(df)

    assert result["customer_id"].tolist() == [
        "CUST100001",
        "CUST100002",
        "CUST100003",
    ]


def test_names_are_trimmed_but_not_uppercased():

    df = pd.DataFrame(
        {
            "name_ar": [
                " أحمد محمد ",
                "محمد علي",
            ]
        }
    )

    result = normalize_dataframe(df)

    assert result["name_ar"].tolist() == [
        "أحمد محمد",
        "محمد علي",
    ]


def test_unknown_categorical_value_is_preserved():

    df = pd.DataFrame(
        {
            "wallet_status": [
                " unexpected_value "
            ]
        }
    )

    result = normalize_dataframe(df)

    assert result["wallet_status"].iloc[0] == (
        "UNEXPECTED_VALUE"
    )


def test_normalization_does_not_modify_original_dataframe():

    df = pd.DataFrame(
        {
            "wallet_status": [" OPENED "]
        }
    )

    result = normalize_dataframe(df)

    assert df["wallet_status"].iloc[0] == " OPENED "

    assert result["wallet_status"].iloc[0] == "OPENED"
def test_unrecognized_boolean_value_raises():
    """
    Boolean-like fields must fail loudly on unrecognized values.
    Unlike categorical fields (which intentionally preserve
    unknown values), a boolean field with a value outside
    TRUE_VALUES/FALSE_VALUES almost always means bad or malformed
    source data (e.g. a typo). Silently passing it through would
    let corrupted data reach the controls undetected.
    """

    df = pd.DataFrame(
        {
            "screening_evidence_present": [
                "maybe",
            ]
        }
    )

    with pytest.raises(ValueError):
        normalize_dataframe(df)    