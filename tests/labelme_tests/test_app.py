import os.path as osp
import shutil
import tempfile

import pytest

import app
import libs.config
import libs.testing

here = osp.dirname(osp.abspath(__file__))
data_dir = osp.join(here, "data")


def _win_show_and_wait_imageData(qtbot, win):
    win.show()

    def check_imageData():
        assert hasattr(win, "imageData")
        assert win.imageData is not None

    qtbot.waitUntil(check_imageData)  # wait for loadFile


@pytest.mark.gui
def test_MainWindow_open(qtbot):
    win = app.MainWindow()
    qtbot.addWidget(win)
    win.show()
    win.close()


@pytest.mark.gui
def test_MainWindow_open_img(qtbot):
    img_file = osp.join(data_dir, "raw/2011_000003.jpg")
    win = app.MainWindow(filename=img_file)
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    win.close()


@pytest.mark.gui
def test_MainWindow_open_json(qtbot):
    json_files = [
        osp.join(data_dir, "annotated_with_data/apc2016_obj3.json"),
        osp.join(data_dir, "annotated/2011_000003.json"),
    ]
    for json_file in json_files:
        libs.testing.assert_labelfile_sanity(json_file)

        win = app.MainWindow(filename=json_file)
        qtbot.addWidget(win)
        _win_show_and_wait_imageData(qtbot, win)
        win.close()


def create_MainWindow_with_directory(qtbot):
    directory = osp.join(data_dir, "raw")
    win = app.MainWindow(filename=directory)
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)
    return win


@pytest.mark.gui
def test_MainWindow_openNextImg(qtbot):
    win = create_MainWindow_with_directory(qtbot)
    win.openNextImg()


@pytest.mark.gui
def test_MainWindow_openPrevImg(qtbot):
    win = create_MainWindow_with_directory(qtbot)
    win.openNextImg()


@pytest.mark.gui
def test_MainWindow_annotate_jpg(qtbot):
    tmp_dir = tempfile.mkdtemp()
    input_file = osp.join(data_dir, "raw/2011_000003.jpg")
    out_file = osp.join(tmp_dir, "2011_000003.json")

    config = libs.config.get_default_config()
    win = app.MainWindow(
        config=config,
        filename=input_file,
        output_file=out_file,
    )
    qtbot.addWidget(win)
    _win_show_and_wait_imageData(qtbot, win)

    label = "whole"
    points = [
        (100, 100),
        (100, 238),
        (400, 238),
        (400, 100),
    ]
    shapes = [
        dict(
            label=label,
            group_id=None,
            points=points,
            shape_type="polygon",
            mask=None,
            flags={},
            other_data={},
        )
    ]
    win.loadLabels(shapes)
    win.saveFile()

    libs.testing.assert_labelfile_sanity(out_file)
    shutil.rmtree(tmp_dir)
