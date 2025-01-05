# -*- encoding: utf-8 -*-

import pytest

from libs.widgets import LabelListWidget
from libs.widgets import LabelListWidgetItem


@pytest.mark.gui
def test_LabelListWidget(qtbot):
    widget = LabelListWidget()

    item = LabelListWidgetItem(text="person <font color='red'>●</fon>")
    widget.addItem(item)
    item = LabelListWidgetItem(text="dog <font color='blue'>●</fon>")
    widget.addItem(item)

    widget.show()
    qtbot.addWidget(widget)
    qtbot.waitExposed(widget)
