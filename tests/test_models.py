"""Tests for wordbatch models dimension mismatch handling."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix


class TestFTRL:
    """Tests for FTRL model."""

    def test_predict_raises_on_dimension_mismatch(self):
        """Test that predict raises ValueError when input dimensions don't match."""
        from wordbatch.models import FTRL

        X_train = csr_matrix(np.random.rand(10, 100))
        y_train = np.random.randint(0, 2, 10).astype(np.float64)

        model = FTRL()
        model.fit(X_train, y_train)

        X_wrong = csr_matrix(np.random.rand(5, 50))

        with pytest.raises(ValueError, match="Dimension mismatch"):
            model.predict(X_wrong)

    def test_fit_raises_on_dimension_mismatch_without_reset(self):
        """Test that fit raises ValueError when dimensions don't match and reset=False."""
        from wordbatch.models import FTRL

        X_train = csr_matrix(np.random.rand(10, 100))
        y_train = np.random.randint(0, 2, 10).astype(np.float64)

        model = FTRL()
        model.fit(X_train, y_train)

        X_wrong = csr_matrix(np.random.rand(5, 50))
        y_wrong = np.random.randint(0, 2, 5).astype(np.float64)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            model.fit(X_wrong, y_wrong, reset=False)



class TestFTRL32:
    """Tests for FTRL32 model."""

    def test_predict_raises_on_dimension_mismatch(self):
        """Test that predict raises ValueError when input dimensions don't match."""
        from wordbatch.models import FTRL32

        X_train = csr_matrix(np.random.rand(10, 100))
        y_train = np.random.randint(0, 2, 10).astype(np.float64)

        model = FTRL32()
        model.fit(X_train, y_train)

        X_wrong = csr_matrix(np.random.rand(5, 50))

        with pytest.raises(ValueError, match="Dimension mismatch"):
            model.predict(X_wrong)

    def test_fit_raises_on_dimension_mismatch_without_reset(self):
        """Test that fit raises ValueError when dimensions don't match and reset=False."""
        from wordbatch.models import FTRL32

        X_train = csr_matrix(np.random.rand(10, 100))
        y_train = np.random.randint(0, 2, 10).astype(np.float64)

        model = FTRL32()
        model.fit(X_train, y_train)

        X_wrong = csr_matrix(np.random.rand(5, 50))
        y_wrong = np.random.randint(0, 2, 5).astype(np.float64)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            model.fit(X_wrong, y_wrong, reset=False)



class TestFMFTRL:
    """Tests for FM_FTRL model."""

    def test_predict_raises_on_dimension_mismatch(self):
        """Test that predict raises ValueError when input dimensions don't match."""
        from wordbatch.models import FM_FTRL

        X_train = csr_matrix(np.random.rand(10, 100))
        y_train = np.random.randint(0, 2, 10).astype(np.float64)

        model = FM_FTRL()
        model.fit(X_train, y_train)

        X_wrong = csr_matrix(np.random.rand(5, 50))

        with pytest.raises(ValueError, match="Dimension mismatch"):
            model.predict(X_wrong)

    def test_fit_raises_on_dimension_mismatch_without_reset(self):
        """Test that fit raises ValueError when dimensions don't match and reset=False."""
        from wordbatch.models import FM_FTRL

        X_train = csr_matrix(np.random.rand(10, 100))
        y_train = np.random.randint(0, 2, 10).astype(np.float64)

        model = FM_FTRL()
        model.fit(X_train, y_train)

        X_wrong = csr_matrix(np.random.rand(5, 50))
        y_wrong = np.random.randint(0, 2, 5).astype(np.float64)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            model.fit(X_wrong, y_wrong, reset=False)


class TestNNReLUH1:
    """Tests for NN_ReLU_H1 model."""

    def test_predict_raises_on_dimension_mismatch(self):
        """Test that predict raises ValueError when input dimensions don't match."""
        from wordbatch.models import NN_ReLU_H1

        X_train = csr_matrix(np.random.rand(10, 100))
        y_train = np.random.randint(0, 2, 10).astype(np.float64)

        model = NN_ReLU_H1()
        model.fit(X_train, y_train)

        X_wrong = csr_matrix(np.random.rand(5, 50))

        with pytest.raises(ValueError, match="Dimension mismatch"):
            model.predict(X_wrong)

    def test_fit_raises_on_dimension_mismatch_without_reset(self):
        """Test that fit raises ValueError when dimensions don't match and reset=False."""
        from wordbatch.models import NN_ReLU_H1

        X_train = csr_matrix(np.random.rand(10, 100))
        y_train = np.random.randint(0, 2, 10).astype(np.float64)

        model = NN_ReLU_H1()
        model.fit(X_train, y_train)

        X_wrong = csr_matrix(np.random.rand(5, 50))
        y_wrong = np.random.randint(0, 2, 5).astype(np.float64)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            model.fit(X_wrong, y_wrong, reset=False)


class TestNNReLUH2:
    """Tests for NN_ReLU_H2 model."""

    def test_predict_raises_on_dimension_mismatch(self):
        """Test that predict raises ValueError when input dimensions don't match."""
        from wordbatch.models import NN_ReLU_H2

        X_train = csr_matrix(np.random.rand(10, 100))
        y_train = np.random.randint(0, 2, 10).astype(np.float64)

        model = NN_ReLU_H2()
        model.fit(X_train, y_train)

        X_wrong = csr_matrix(np.random.rand(5, 50))

        with pytest.raises(ValueError, match="Dimension mismatch"):
            model.predict(X_wrong)

    def test_fit_raises_on_dimension_mismatch_without_reset(self):
        """Test that fit raises ValueError when dimensions don't match and reset=False."""
        from wordbatch.models import NN_ReLU_H2

        X_train = csr_matrix(np.random.rand(10, 100))
        y_train = np.random.randint(0, 2, 10).astype(np.float64)

        model = NN_ReLU_H2()
        model.fit(X_train, y_train)

        X_wrong = csr_matrix(np.random.rand(5, 50))
        y_wrong = np.random.randint(0, 2, 5).astype(np.float64)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            model.fit(X_wrong, y_wrong, reset=False)

