"""Synthetic disclosure revisions; no network access or persistent writes."""
from usmd.financial_normalize import normalize_company
from usmd.financial_query import select_values


def main():
    issuer = {
        "cik": "123", "issuer_id": "SEC:0000000123", "symbol": "EXAMPLE",
        "companyfacts_source_id": "synthetic:two-revisions",
    }
    common = {
        "start": "2024-01-01", "end": "2024-03-31", "form": "10-Q",
        "fy": 2024, "fp": "Q1",
    }
    body = {
        "cik": 123, "entityName": "Synthetic Example Company",
        "facts": {"us-gaap": {"Revenues": {"units": {"USD": [
            {**common, "val": 100, "filed": "2024-05-01", "accn": "original"},
            {**common, "val": 110, "filed": "2024-09-15", "accn": "revision", "form": "10-Q/A"},
        ]}}}},
    }
    values = normalize_company(body, [], issuer, "2024-10-01T00:00:00Z")["financial_values"]
    print("Synthetic disclosure versions:")
    print(values[["metric_id", "value", "filed", "accn"]].to_string(index=False))
    for as_of in ("2024-06-01", "2024-10-01"):
        result = select_values(values, metrics="revenue", as_of=as_of)
        expected = 100 if as_of == "2024-06-01" else 110
        assert result.value.tolist() == [expected], result
        print(f"as_of={as_of}: revenue={result.iloc[0].value:g} USD")
    print("These are conservative daily availability estimates, not exact intraday publication times.")


if __name__ == "__main__":
    main()
