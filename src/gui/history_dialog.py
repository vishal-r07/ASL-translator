#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
History Dialog

This module provides a dialog for viewing and managing translation history.
"""

import os
import sys
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QFileDialog, QMessageBox, QLineEdit,
    QComboBox, QDateEdit, QCheckBox, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QDate
from PyQt5.QtGui import QIcon, QFont

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility functions and history manager
from src.history.manager import HistoryManager
from src.utils.common import get_assets_dir


class HistoryDialog(QDialog):
    """
    Dialog for viewing and managing translation history.
    """
    
    def __init__(self, history_manager: HistoryManager, parent=None):
        """
        Initialize the history dialog.
        
        Args:
            history_manager: The history manager instance.
            parent: Parent widget.
        """
        super().__init__(parent)
        
        self.setWindowTitle("Translation History")
        self.setMinimumSize(800, 600)
        
        self.history_manager = history_manager
        self.history_entries = []
        self.filtered_entries = []
        
        # Initialize UI
        self._init_ui()
        
        # Load history entries
        self._load_history()
    
    def _init_ui(self):
        """
        Initialize the user interface.
        """
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create search and filter group
        search_group = QGroupBox("Search and Filter")
        search_layout = QVBoxLayout(search_group)
        
        # Search form layout
        form_layout = QFormLayout()
        
        # Search by text
        self.search_text = QLineEdit()
        self.search_text.setPlaceholderText("Search by text...")
        self.search_text.textChanged.connect(self._apply_filters)
        form_layout.addRow("Search Text:", self.search_text)
        
        # Filter by type
        self.filter_type = QComboBox()
        self.filter_type.addItems(["All", "ASL", "Emotion"])
        self.filter_type.currentIndexChanged.connect(self._apply_filters)
        form_layout.addRow("Type:", self.filter_type)
        
        # Filter by date range
        date_range_layout = QHBoxLayout()
        
        self.filter_date_from = QDateEdit()
        self.filter_date_from.setCalendarPopup(True)
        self.filter_date_from.setDate(QDate.currentDate().addMonths(-1))
        date_range_layout.addWidget(self.filter_date_from)
        
        date_range_layout.addWidget(QLabel("to"))
        
        self.filter_date_to = QDateEdit()
        self.filter_date_to.setCalendarPopup(True)
        self.filter_date_to.setDate(QDate.currentDate())
        date_range_layout.addWidget(self.filter_date_to)
        
        self.use_date_filter = QCheckBox("Enable Date Filter")
        self.use_date_filter.setChecked(False)
        self.use_date_filter.stateChanged.connect(self._apply_filters)
        date_range_layout.addWidget(self.use_date_filter)
        
        form_layout.addRow("Date Range:", date_range_layout)
        
        # Add form layout to search layout
        search_layout.addLayout(form_layout)
        
        # Add search group to main layout
        main_layout.addWidget(search_group)
        
        # Create table widget for history entries
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["Timestamp", "Type", "Text", "Confidence"])
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_widget.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table_widget.verticalHeader().setVisible(False)
        
        # Add table widget to main layout
        main_layout.addWidget(self.table_widget)
        
        # Create buttons layout
        buttons_layout = QHBoxLayout()
        
        # Export button
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self._export_history)
        buttons_layout.addWidget(self.export_button)
        
        # Clear button
        self.clear_button = QPushButton("Clear History")
        self.clear_button.clicked.connect(self._clear_history)
        buttons_layout.addWidget(self.clear_button)
        
        # Close button
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        buttons_layout.addWidget(self.close_button)
        
        # Add buttons layout to main layout
        main_layout.addLayout(buttons_layout)
    
    def _load_history(self):
        """
        Load history entries from the history manager.
        """
        self.history_entries = self.history_manager.get_all_entries()
        self._apply_filters()
    
    def _apply_filters(self):
        """
        Apply filters to history entries.
        """
        # Get filter values
        search_text = self.search_text.text().lower()
        filter_type = self.filter_type.currentText()
        use_date_filter = self.use_date_filter.isChecked()
        date_from = self.filter_date_from.date().toPyDate() if use_date_filter else None
        date_to = self.filter_date_to.date().toPyDate() if use_date_filter else None
        
        # Apply filters
        self.filtered_entries = []
        for entry in self.history_entries:
            # Filter by text
            if search_text and search_text not in entry["text"].lower():
                continue
            
            # Filter by type
            if filter_type != "All" and entry["type"] != filter_type.lower():
                continue
            
            # Filter by date range
            if use_date_filter:
                entry_date = datetime.fromisoformat(entry["timestamp"]).date()
                if entry_date < date_from or entry_date > date_to:
                    continue
            
            # Add entry to filtered entries
            self.filtered_entries.append(entry)
        
        # Update table widget
        self._update_table()
    
    def _update_table(self):
        """
        Update the table widget with filtered entries.
        """
        # Clear table
        self.table_widget.setRowCount(0)
        
        # Add filtered entries to table
        for i, entry in enumerate(self.filtered_entries):
            self.table_widget.insertRow(i)
            
            # Format timestamp
            timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            
            # Add items to row
            self.table_widget.setItem(i, 0, QTableWidgetItem(timestamp))
            self.table_widget.setItem(i, 1, QTableWidgetItem(entry["type"].upper()))
            self.table_widget.setItem(i, 2, QTableWidgetItem(entry["text"]))
            self.table_widget.setItem(i, 3, QTableWidgetItem(f"{entry['confidence']:.2f}"))
    
    def _export_history(self):
        """
        Export history entries to a file.
        """
        # Ask for file path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export History",
            os.path.expanduser("~/Documents/asl_translator_history.csv"),
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Export history
        try:
            success = self.history_manager.export_to_csv(file_path, self.filtered_entries)
            
            if success:
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"History exported to {file_path}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Export Failed",
                    "Failed to export history"
                )
        
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Error exporting history: {str(e)}"
            )
    
    def _clear_history(self):
        """
        Clear all history entries.
        """
        # Confirm clear
        reply = QMessageBox.question(
            self,
            "Clear History",
            "Are you sure you want to clear all history entries?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clear history
            self.history_manager.clear()
            
            # Reload history
            self._load_history()
            
            # Show confirmation message
            QMessageBox.information(
                self,
                "History Cleared",
                "All history entries have been cleared"
            )