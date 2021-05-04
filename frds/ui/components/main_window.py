"""MainWindow class"""

from importlib.resources import open_text
from PyQt5.QtCore import QUrl
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QMessageBox, QFileSystemModel
from frds.settings import FRDS_HOME_PAGE
import frds.ui.designs
from frds.ui.components import Preferences, TreeViewMeasures
from frds.utils.settings import get_root_dir
from frds.multiprocessing.threads import ThreadsManager, ThreadWorker
import frds.measures

ui = open_text(frds.ui.designs, "MainWindow.ui")


class MainWindow(*uic.loadUiType(ui)):
    def __init__(self):
        super().__init__()
        super().setupUi(self)
        self.threadpool = ThreadsManager(self)

        # Preference settings
        self.pref_window = Preferences(self)

        # File explorer
        self.filesystermModel = QFileSystemModel()
        self.filesystermModel.setRootPath(get_root_dir())
        self.treeViewFilesystem.setModel(self.filesystermModel)
        self.treeViewFilesystem.setRootIndex(
            self.filesystermModel.index(get_root_dir())
        )
        # Setup treeView of corporate finance measures
        self.treeViewCorpFinc = TreeViewMeasures(self)
        self.tabCorpFinc.layout().addWidget(self.treeViewCorpFinc)
        self.treeViewCorpFinc.addMeasures(
            frds.measures.corporate_finance,
            self.treeViewCorpFinc.model.invisibleRootItem(),
        )
        self.treeViewCorpFinc.expandAll()
        # Setup treeView of banking measures
        self.treeViewBanking = TreeViewMeasures(self)
        self.tabBanking.layout().addWidget(self.treeViewBanking)
        self.treeViewCorpFinc.addMeasures(
            frds.measures.banking,
            self.treeViewBanking.model.invisibleRootItem(),
        )
        self.treeViewBanking.expandAll()
        # Setup treeView of market microstructure measures
        self.treeViewMktStructure = TreeViewMeasures(self)
        self.tabMktStructure.layout().addWidget(self.treeViewMktStructure)
        self.treeViewMktStructure.addMeasures(
            frds.measures.market_microstructure,
            self.treeViewMktStructure.model.invisibleRootItem(),
        )
        self.treeViewMktStructure.expandAll()

        # Tabify dock widgets
        self.tabifyDockWidget(self.dockWidgetFilesystem, self.dockWidgetHistory)
        self.dockWidgetFilesystem.raise_()
        # Connect signals
        self.actionAbout_Qt.triggered.connect(lambda: QMessageBox.aboutQt(self))
        self.actionRestoreViews.triggered.connect(self.restoreAllViews)
        self.actionFile_Explorer.triggered.connect(self.toggleFileExplorer)
        self.actionPreferences.triggered.connect(self.pref_window.show)
        self.actionDocumentation.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(FRDS_HOME_PAGE))
        )
        self.threadpool.status.connect(self.statusbar.showMessage)

    def restoreAllViews(self):
        self.dockWidgetFilesystem.show()
        self.dockWidgetFilesystem.setFloating(False)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockWidgetFilesystem)

    def toggleFileExplorer(self):
        if self.actionFile_Explorer.isChecked():
            self.dockWidgetFilesystem.show()
        else:
            self.dockWidgetFilesystem.hide()
