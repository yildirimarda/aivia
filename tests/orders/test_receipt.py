"""A simple receipt carries the line item, currency and one flat tax estimate.

The receipt ends with a footer stating that it is not a tax invoice.
"""

from __future__ import annotations

from aivia.core.config.orders import OrderPricingConfig
from aivia.orders import export_receipt


def _pricing(
    *,
    currency: str = "USD",
    amount: float = 0.0,
    tax_rate: float = 0.0,
    tax_label: str = "Estimated tax",
) -> OrderPricingConfig:
    return OrderPricingConfig(
        currency=currency,
        amount=amount,
        tax_rate=tax_rate,
        tax_label=tax_label,
    )


def test_receipt_contains_the_line_item() -> None:
    receipt = export_receipt("Birthday card for Ada", _pricing(amount=10.0))

    assert "Birthday card for Ada" in receipt


def test_receipt_contains_the_currency() -> None:
    receipt = export_receipt("Card", _pricing(currency="EUR", amount=10.0))

    assert "EUR" in receipt


def test_receipt_contains_the_flat_tax_estimate() -> None:
    receipt = export_receipt("Card", _pricing(amount=100.0, tax_rate=0.10))

    assert "10.00" in receipt  # 100.00 * 0.10


def test_receipt_contains_the_not_a_tax_invoice_footer() -> None:
    receipt = export_receipt("Card", _pricing())

    assert "This is not a tax invoice." in receipt


def test_receipt_defaults_to_zero_amount_and_zero_tax() -> None:
    receipt = export_receipt("Card", _pricing())

    assert "Amount: USD 0.00" in receipt
    assert "Estimated tax: USD 0.00" in receipt
    assert "Total: USD 0.00" in receipt


def test_receipt_shows_amount_tax_and_total() -> None:
    receipt = export_receipt("Card", _pricing(amount=50.0, tax_rate=0.20))

    assert "Amount: USD 50.00" in receipt
    assert "Estimated tax: USD 10.00" in receipt
    assert "Total: USD 60.00" in receipt


def test_receipt_uses_the_configured_tax_label() -> None:
    receipt = export_receipt("Card", _pricing(tax_label="VAT"))

    assert "VAT:" in receipt


def test_export_receipt_returns_a_single_string() -> None:
    receipt = export_receipt("Card", _pricing())

    assert isinstance(receipt, str)
    assert receipt.count("This is not a tax invoice.") == 1
