#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text-to-Speech Utility for ASL Translator and Emotion Communicator

This module provides functions for text-to-speech conversion.
"""

import os
import threading
import queue
import time

# Try to import pyttsx3, but don't fail if it's not available
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("pyttsx3 not available. Text-to-speech functionality will be disabled.")


class TextToSpeech:
    """Text-to-Speech class for converting text to speech."""
    
    def __init__(self, rate=150, volume=1.0, voice=None):
        """
        Initialize the text-to-speech engine.
        
        Args:
            rate (int): Speech rate (words per minute).
            volume (float): Volume level (0.0 to 1.0).
            voice (str, optional): Voice ID to use. If None, the default voice is used.
        """
        self.rate = rate
        self.volume = volume
        self.voice = voice
        self.engine = None
        self.available = False
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.stop_requested = False
        self.speech_thread = None
        
        # Initialize the engine if pyttsx3 is available
        if PYTTSX3_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', self.rate)
                self.engine.setProperty('volume', self.volume)
                
                # Set voice if specified
                if self.voice is not None:
                    self.engine.setProperty('voice', self.voice)
                
                self.available = True
                
                # Start the speech thread
                self.speech_thread = threading.Thread(target=self._speech_worker)
                self.speech_thread.daemon = True
                self.speech_thread.start()
                
                print("Text-to-speech engine initialized successfully.")
            except Exception as e:
                print(f"Error initializing text-to-speech engine: {e}")
    
    def _speech_worker(self):
        """
        Worker thread for processing speech requests from the queue.
        """
        while True:
            if self.stop_requested:
                break
            
            try:
                # Get a text item from the queue with a timeout
                text = self.speech_queue.get(timeout=0.5)
                
                # Speak the text
                self.is_speaking = True
                self.engine.say(text)
                self.engine.runAndWait()
                self.is_speaking = False
                
                # Mark the task as done
                self.speech_queue.task_done()
            except queue.Empty:
                # Queue is empty, just continue
                pass
            except Exception as e:
                print(f"Error in speech worker: {e}")
                self.is_speaking = False
    
    def speak(self, text):
        """
        Speak the given text.
        
        Args:
            text (str): The text to speak.
            
        Returns:
            bool: True if the text was added to the speech queue, False otherwise.
        """
        if not self.available or not text:
            return False
        
        try:
            # Add the text to the speech queue
            self.speech_queue.put(text)
            return True
        except Exception as e:
            print(f"Error adding text to speech queue: {e}")
            return False
    
    def stop(self):
        """
        Stop the current speech and clear the queue.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.available:
            return False
        
        try:
            # Stop the current speech
            self.engine.stop()
            
            # Clear the queue
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                    self.speech_queue.task_done()
                except queue.Empty:
                    break
            
            self.is_speaking = False
            return True
        except Exception as e:
            print(f"Error stopping speech: {e}")
            return False
    
    def set_rate(self, rate):
        """
        Set the speech rate.
        
        Args:
            rate (int): Speech rate (words per minute).
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.available:
            return False
        
        try:
            self.rate = rate
            self.engine.setProperty('rate', rate)
            return True
        except Exception as e:
            print(f"Error setting speech rate: {e}")
            return False
    
    def set_volume(self, volume):
        """
        Set the speech volume.
        
        Args:
            volume (float): Volume level (0.0 to 1.0).
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.available:
            return False
        
        try:
            self.volume = volume
            self.engine.setProperty('volume', volume)
            return True
        except Exception as e:
            print(f"Error setting speech volume: {e}")
            return False
    
    def set_voice(self, voice):
        """
        Set the speech voice.
        
        Args:
            voice (str): Voice ID to use.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.available:
            return False
        
        try:
            self.voice = voice
            self.engine.setProperty('voice', voice)
            return True
        except Exception as e:
            print(f"Error setting speech voice: {e}")
            return False
    
    def get_available_voices(self):
        """
        Get a list of available voices.
        
        Returns:
            list: A list of available voice IDs, or an empty list if not available.
        """
        if not self.available:
            return []
        
        try:
            voices = self.engine.getProperty('voices')
            return [voice.id for voice in voices]
        except Exception as e:
            print(f"Error getting available voices: {e}")
            return []
    
    def is_available(self):
        """
        Check if text-to-speech is available.
        
        Returns:
            bool: True if available, False otherwise.
        """
        return self.available
    
    def is_currently_speaking(self):
        """
        Check if the engine is currently speaking.
        
        Returns:
            bool: True if speaking, False otherwise.
        """
        return self.is_speaking
    
    def shutdown(self):
        """
        Shutdown the text-to-speech engine.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.available:
            return False
        
        try:
            # Stop any current speech
            self.stop()
            
            # Signal the speech thread to stop
            self.stop_requested = True
            
            # Wait for the speech thread to finish (with timeout)
            if self.speech_thread is not None and self.speech_thread.is_alive():
                self.speech_thread.join(timeout=2.0)
            
            # Clean up the engine
            self.engine = None
            self.available = False
            
            return True
        except Exception as e:
            print(f"Error shutting down text-to-speech engine: {e}")
            return False
    
    def __del__(self):
        """
        Destructor to ensure the engine is properly shut down.
        """
        self.shutdown()