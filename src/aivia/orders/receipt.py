"""Simple order receipts: line item, currency and one flat tax estimate.

Receipts are not tax engines: they carry one flat tax rate and state that they
are not tax invoices (decision D-011).
"""

from __future__ import annotations

from aivia.core.config.orders import OrderPricingConfig


class Receipt:
    """A simple receipt: a line item, currency and a single flat tax estimate."""

    def __init__(self, line_item: str, pricing: OrderPricingConfig) -> None:
        self.line_item = line_item
        self.pricing = pricing

    @property
    def tax_amount(self) -> float:
        """The flat tax estimate for the line-item amount."""
        return self.pricing.tax_amount

    @property
    def total(self) -> float:
        """Line-item amount plus the flat tax estimate."""
        return self.pricing.amount + self.tax_amount

    def render(self) -> str:
        """Render the receipt as plain text."""
        currency = self.pricing.currency_code
        lines = [
            "Receipt",
            "-------",
            f"Line item: {self.line_item}",
            f"Amount: {currency} {self.pricing.amount:.2f}",
            f"{self.pricing.tax_label}: {currency} {self.tax_amount:.2f}",
            f"Total: {currency} {self.total:.2f}",
            "",
            "This is not a tax invoice.",
        ]
        return "\n".join(lines)


def export_receipt(line_item: str, pricing: OrderPricingConfig) -> str:
    """Export a simple receipt for *line_item* using *pricing*.

    The receipt carries the line item, currency and one flat tax estimate and
    ends with a footer stating that it is not a tax invoice.
    """
    return Receipt(line_item, pricing).render()
