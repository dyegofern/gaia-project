"""Tests for chess vision improvements."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

from gaia_agent.chess_vision import (
    classify_square,
    CONFIDENCE_THRESHOLD,
    load_templates,
)


class TestChessVisionConfidence:
    def test_confidence_threshold_constant(self):
        """Test that CONFIDENCE_THRESHOLD is set to 0.2."""
        assert CONFIDENCE_THRESHOLD == 0.2

    def test_low_iu_return_none(self):
        """Test that low IoU scores return None (empty square)."""
        templates = load_templates()

        # Create a mock square that will have very low IoU with all templates
        mock_square = MagicMock()

        with patch("gaia_agent.chess_vision._square_mask") as mock_mask, \
             patch("gaia_agent.chess_vision._normalized_silhouette") as mock_sil, \
             patch("gaia_agent.chess_vision._piece_colour") as mock_colour:

            # Mock low mean mask (below EMPTY_SQUARE_MAX_FILL)
            mock_mask.return_value = np.zeros((48, 48), dtype=bool)

            result = classify_square(mock_square, templates)

            assert result is None

    def test_iu_below_threshold_returns_none(self):
        """Test that IoU below CONFIDENCE_THRESHOLD returns None."""
        templates = load_templates()

        mock_square = MagicMock()

        with patch("gaia_agent.chess_vision._square_mask") as mock_mask, \
             patch("gaia_agent.chess_vision._normalized_silhouette") as mock_sil, \
             patch("gaia_agent.chess_vision._piece_colour") as mock_colour:

            # Mock mask with some content
            mask = np.zeros((48, 48), dtype=bool)
            mask[20:28, 20:28] = True
            mock_mask.return_value = mask

            # Mock silhouette that will have low IoU
            silhouette = np.zeros((48, 48), dtype=bool)
            silhouette[0:4, 0:4] = True  # Different region

            mock_sil.return_value = silhouette
            mock_colour.return_value = "w"

            # Mock templates with completely different pieces
            fake_templates = {
                "p": np.ones((48, 48), dtype=bool),  # All true
                "r": np.ones((48, 48), dtype=bool),
            }

            result = classify_square(mock_square, fake_templates)

            # IoU should be very low, returning None
            assert result is None

    def test_iu_above_threshold_returns_piece(self):
        """Test that IoU above CONFIDENCE_THRESHOLD returns piece type."""
        templates = load_templates()

        mock_square = MagicMock()

        with patch("gaia_agent.chess_vision._square_mask") as mock_mask, \
             patch("gaia_agent.chess_vision._normalized_silhouette") as mock_sil, \
             patch("gaia_agent.chess_vision._piece_colour") as mock_colour:

            # Mock mask with content
            mask = np.zeros((48, 48), dtype=bool)
            mask[15:33, 15:33] = True
            mock_mask.return_value = mask

            # Mock silhouette similar to template
            silhouette = np.zeros((48, 48), dtype=bool)
            silhouette[15:33, 15:33] = True  # Same region

            mock_sil.return_value = silhouette
            mock_colour.return_value = "w"

            # Mock templates
            fake_templates = {
                "p": silhouette,  # Perfect match
                "r": np.ones((48, 48), dtype=bool),
            }

            result = classify_square(mock_square, fake_templates)

            # Should return the pawn (white)
            assert result == "P"


class TestChessVisionIntegration:
    def test_classify_square_with_mock_templates(self):
        """Test classify_square with simulated template matching."""
        templates = load_templates()

        # Create a mock square
        mock_square = MagicMock()

        # Create a mask with moderate content (above EMPTY_SQUARE_MAX_FILL)
        mask = np.zeros((48, 48), dtype=bool)
        mask[10:38, 10:38] = True

        with patch("gaia_agent.chess_vision._square_mask", return_value=mask), \
             patch("gaia_agent.chess_vision._normalized_silhouette") as mock_sil, \
             patch("gaia_agent.chess_vision._piece_colour", return_value="b"):

            # Create silhouette with good match to one template
            test_silhouette = np.zeros((48, 48), dtype=bool)
            test_silhouette[12:36, 12:36] = True
            mock_sil.return_value = test_silhouette

            result = classify_square(mock_square, templates)

            # Should return a piece (lowercase for black) or None if no good match
            assert result is None or (isinstance(result, str) and result.islower())
