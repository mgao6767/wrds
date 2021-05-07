"""MainWindow class"""

from importlib.resources import open_text, read_binary
from PyQt5 import uic
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QPixmap, QIcon
from PyQt5.QtWidgets import QMessageBox, QFileSystemModel, QTreeWidgetItem
from frds.settings import FRDS_HOME_PAGE
import frds.ui.designs
import frds.ui.resources
from frds.ui.components import Preferences, TreeViewMeasures, Documentation
from frds.utils.settings import get_root_dir
from frds.multiprocessing.threads import ThreadWorker, ThreadsManager
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
        self.treeViewCorpFinc.addMeasures(frds.measures.corporate_finance)
        self.treeViewCorpFinc.expandAll()
        # Setup treeView of banking measures
        self.treeViewBanking = TreeViewMeasures(self)
        self.tabBanking.layout().addWidget(self.treeViewBanking)
        self.treeViewBanking.addMeasures(frds.measures.banking)
        self.treeViewBanking.expandAll()
        # Setup treeView of market microstructure measures
        self.treeViewMktStructure = TreeViewMeasures(self)
        self.tabMktStructure.layout().addWidget(self.treeViewMktStructure)
        self.treeViewMktStructure.addMeasures(
            frds.measures.market_microstructure)
        self.treeViewMktStructure.expandAll()
        # Setup webview of documentation
        self.documentation_webview = Documentation(self)
        self.dockWidgetDocumentationContents.layout().addWidget(self.documentation_webview)
        self.restoreAllViews()

        # ToolBar
        self.__setup_toolBar()
        # Connect signals
        self.__connect_signals()
        # Start background tasks
        self.__start_background_tasks()

    def __connect_signals(self):
        # Connect signals
        self.actionSettings.triggered.connect(self.pref_window.show)
        self.actionAbout_Qt.triggered.connect(
            lambda: QMessageBox.aboutQt(self))
        self.actionRestoreViews.triggered.connect(self.restoreAllViews)
        self.actionFile_Explorer.triggered.connect(
            lambda: self.__show(self.dockWidgetFilesystem))
        self.actionDocumentation.triggered.connect(
            lambda: self.__show(self.dockWidgetDocumentation))
        self.actionPreferences.triggered.connect(self.pref_window.show)
        self.actionAbout.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(FRDS_HOME_PAGE)))
        self.threadpool.status.connect(self.statusbar.showMessage)
        self.treeViewCorpFinc.itemClicked.connect(self.__measure_selected)
        self.treeViewBanking.itemClicked.connect(self.__measure_selected)
        self.treeViewMktStructure.itemClicked.connect(self.__measure_selected)
        self.treeViewCorpFinc.itemDoubleClicked.connect(
            self.__measure_double_clicked)

    def restoreAllViews(self):
        self.dockWidgetFilesystem.show()
        self.dockWidgetFilesystem.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dockWidgetFilesystem)
        self.dockWidgetDocumentation.show()
        self.dockWidgetDocumentation.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea,
                           self.dockWidgetDocumentation)
        # Tabify dock widgets
        self.tabifyDockWidget(
            self.dockWidgetDocumentation, self.dockWidgetFilesystem)
        self.dockWidgetDocumentation.raise_()

    def __show(self, widget):
        widget.show()
        widget.raise_()

    def __start_background_tasks(self):
        # worker = ThreadWorker("Generate local docs",
        #                       self.__generate_local_docs)
        # worker.signals.finished.connect(self.documentation_webview.displayDoc)
        # self.threadpool.enqueue(worker)
        pass

    def __generate_local_docs(self):
        pass

    def __setup_toolBar(self):
        icn = QPixmap()
        icn.loadFromData(read_binary(frds.ui.resources,
                                     'playback_play_icon&24.png'))
        self.actionRun.setIcon(QIcon(icn))
        icn.loadFromData(read_binary(
            frds.ui.resources, 'round_delete_icon&24.png'))
        self.actionStop.setIcon(QIcon(icn))
        icn.loadFromData(read_binary(frds.ui.resources,
                                     'playback_pause_icon&24.png'))
        self.actionPause.setIcon(QIcon(icn))
        icn.loadFromData(read_binary(frds.ui.resources, 'cogs_icon&24.png'))
        self.actionSettings.setIcon(QIcon(icn))
        icn.loadFromData(read_binary(
            frds.ui.resources, 'dashboard_icon&24.png'))
        self.actionDashboard.setIcon(QIcon(icn))
        icn.loadFromData(read_binary(frds.ui.resources,
                                     'star_fav_empty_icon&24.png'))
        self.actionFavourite.setIcon(QIcon(icn))
        # Add actions to the toolbar
        self.toolBar.addAction(self.actionRun)
        self.toolBar.addAction(self.actionPause)
        self.toolBar.addAction(self.actionStop)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionFavourite)
        self.toolBar.addAction(self.actionDashboard)
        self.toolBar.addAction(self.actionSettings)

    def __measure_selected(self, item: QTreeWidgetItem, column: int):
        if item.columnCount() == 1:
            return
        doc_url = QUrl(item.data(TreeViewMeasures.DocUrl, Qt.DisplayRole))
        current_url = self.documentation_webview.url().url(QUrl.StripTrailingSlash)
        if doc_url.url(QUrl.StripTrailingSlash) != current_url:
            self.documentation_webview.load(QUrl(doc_url))

    def __measure_double_clicked(self, item: QTreeWidgetItem, column: int):
        if item.checkState(TreeViewMeasures.Name) == Qt.Checked:
            item.setCheckState(TreeViewMeasures.Name, Qt.Unchecked)
        elif item.checkState(TreeViewMeasures.Name) == Qt.Unchecked:
            item.setCheckState(TreeViewMeasures.Name, Qt.Checked)
