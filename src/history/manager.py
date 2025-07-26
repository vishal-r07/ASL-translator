#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
History Manager for ASL Translator and Emotion Communicator

This module provides the HistoryManager class for managing translation history.
"""

import os
import json
import datetime


class HistoryManager:
    """History Manager class for managing translation history."""
    
    def __init__(self, max_entries=100, history_file=None):
        """
        Initialize the history manager.
        
        Args:
            max_entries (int): Maximum number of history entries to keep in memory.
            history_file (str): Path to the history file. If None, a default file will be used.
        """
        self.max_entries = max_entries
        self.entries = []
        
        # Set default history file path if not provided
        if history_file is None:
            # Use default history file path in user's documents folder
            documents_dir = os.path.join(os.path.expanduser("~"), "Documents")
            self.history_file = os.path.join(documents_dir, "asl_translator_history.json")
        else:
            self.history_file = history_file
        
        # Load existing history if available
        self.load_from_file()
    
    def add_entry(self, asl_text, emotion=None):
        """
        Add a new entry to the history.
        
        Args:
            asl_text (str): The ASL translation text.
            emotion (str, optional): The detected emotion.
            
        Returns:
            bool: True if the entry was added successfully, False otherwise.
        """
        if not asl_text:
            return False
        
        # Create a new entry
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "asl_text": asl_text,
            "emotion": emotion if emotion else "unknown"
        }
        
        # Add the entry to the list
        self.entries.append(entry)
        
        # Trim the list if it exceeds the maximum number of entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        
        # Save the updated history to file
        self.save_to_file()
        
        return True
    
    def get_entries(self, count=None):
        """
        Get the history entries.
        
        Args:
            count (int, optional): Number of recent entries to retrieve. If None, all entries are returned.
            
        Returns:
            list: A list of history entries.
        """
        if count is None or count >= len(self.entries):
            return self.entries
        
        return self.entries[-count:]
    
    def clear_entries(self):
        """
        Clear all history entries.
        
        Returns:
            bool: True if the entries were cleared successfully, False otherwise.
        """
        self.entries = []
        
        # Save the updated (empty) history to file
        return self.save_to_file()
    
    def save_to_file(self, file_path=None):
        """
        Save the history entries to a file.
        
        Args:
            file_path (str, optional): Path to the file to save to. If None, the default file is used.
            
        Returns:
            bool: True if the entries were saved successfully, False otherwise.
        """
        if file_path is None:
            file_path = self.history_file
        
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the entries to a JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.entries, f, indent=4)
            
            return True
        except Exception as e:
            print(f"Error saving history to file: {e}")
            return False
    
    def load_from_file(self, file_path=None):
        """
        Load history entries from a file.
        
        Args:
            file_path (str, optional): Path to the file to load from. If None, the default file is used.
            
        Returns:
            bool: True if the entries were loaded successfully, False otherwise.
        """
        if file_path is None:
            file_path = self.history_file
        
        if not os.path.exists(file_path):
            return False
        
        try:
            # Load the entries from a JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                self.entries = json.load(f)
            
            # Ensure we don't exceed the maximum number of entries
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries:]
            
            return True
        except Exception as e:
            print(f"Error loading history from file: {e}")
            return False
    
    def export_to_text(self, file_path):
        """
        Export the history entries to a text file.
        
        Args:
            file_path (str): Path to the text file to export to.
            
        Returns:
            bool: True if the entries were exported successfully, False otherwise.
        """
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Export the entries to a text file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("ASL Translator and Emotion Communicator - Translation History\n")
                f.write("=================================================================\n\n")
                
                for entry in self.entries:
                    timestamp = entry["timestamp"]
                    asl_text = entry["asl_text"]
                    emotion = entry["emotion"]
                    
                    f.write(f"[{timestamp}] ASL: {asl_text} | Emotion: {emotion}\n")
            
            return True
        except Exception as e:
            print(f"Error exporting history to text file: {e}")
            return False
    
    def search_entries(self, query):
        """
        Search for entries containing the query string.
        
        Args:
            query (str): The search query.
            
        Returns:
            list: A list of matching history entries.
        """
        if not query:
            return []
        
        # Convert query to lowercase for case-insensitive search
        query = query.lower()
        
        # Search for entries containing the query in ASL text or emotion
        matching_entries = []
        for entry in self.entries:
            if (query in entry["asl_text"].lower() or 
                    query in entry["emotion"].lower()):
                matching_entries.append(entry)
        
        return matching_entries