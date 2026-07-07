import React, { useState, useRef, useEffect } from 'react';

function App() {
  const [brushSize, setBrushSize] = useState(16);
  const [autoPredict, setAutoPredict] = useState(true);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [probabilities, setProbabilities] = useState(new Array(10).fill(0));
  const [preprocessedImage, setPreprocessedImage] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [error, setError] = useState(null);

  const canvasRef = useRef(null);
  const lastPos = useRef({ x: 0, y: 0 });

  // Initialize canvas with white background on mount
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
  }, []);

  const getCoordinates = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    // Check if it is a touch event
    const isTouch = e.touches && e.touches.length > 0;
    const clientX = isTouch ? e.touches[0].clientX : e.clientX;
    const clientY = isTouch ? e.touches[0].clientY : e.clientY;

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  };

  const startDrawing = (e) => {
    e.preventDefault();
    const coords = getCoordinates(e);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.fillStyle = '#000000';
      ctx.beginPath();
      ctx.arc(coords.x, coords.y, brushSize / 2, 0, Math.PI * 2);
      ctx.fill();

      setIsDrawing(true);
      lastPos.current = coords;
      setError(null);
    }
  };

  const draw = (e) => {
    if (!isDrawing) return;
    e.preventDefault();
    const coords = getCoordinates(e);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.lineWidth = brushSize;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.strokeStyle = '#000000';

      ctx.beginPath();
      ctx.moveTo(lastPos.current.x, lastPos.current.y);
      ctx.lineTo(coords.x, coords.y);
      ctx.stroke();

      lastPos.current = coords;
    }
  };

  const stopDrawing = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    if (autoPredict) {
      handlePrediction();
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    setPrediction(null);
    setConfidence(null);
    setProbabilities(new Array(10).fill(0));
    setPreprocessedImage(null);
    setError(null);
  };

  const handlePrediction = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Show scanner effect
    setIsScanning(true);
    setError(null);

    const base64Image = canvas.toDataURL('image/png');

    try {
      const apiUrl = import.meta.env.VITE_API_URL || (
        window.location.port === '5173' 
          ? 'http://127.0.0.1:8000/api/predict' 
          : '/api/predict'
      );

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image_base64: base64Image }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const data = await response.json();
      setPrediction(data.digit);
      setConfidence(data.confidence);
      setProbabilities(data.probabilities);
      setPreprocessedImage(data.preprocessed_image);
    } catch (err) {
      console.error(err);
      setError(err.message || 'Error communicating with model server.');
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <div className="app-container">
      <div className="glow-orb orb-1"></div>
      <div className="glow-orb orb-2"></div>

      <header className="main-header">
        <div className="logo">
          <i className="fa-solid fa-brain logo-icon"></i>
          <h1>Neural Digit Scanner</h1>
        </div>
        <div className="header-badges">
          <span className="badge badge-cnn">CNN Model</span>
          <span className="badge badge-fastapi">FastAPI</span>
        </div>
      </header>

      <main className="dashboard">
        {/* Drawing Panel Card */}
        <section className="card drawing-card" aria-labelledby="drawing-heading">
          <div className="card-header">
            <h2 id="drawing-heading">
              <i className="fa-solid fa-pen-fancy"></i> Drawing Canvas
            </h2>
            <p className="subtitle">Draw any single digit (0-9) inside the grid below.</p>
          </div>

          <div className="canvas-container">
            <canvas
              id="drawing-canvas"
              ref={canvasRef}
              width={280}
              height={280}
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
              onTouchStart={startDrawing}
              onTouchMove={draw}
              onTouchEnd={stopDrawing}
            ></canvas>
          </div>

          <div className="canvas-controls">
            <div className="brush-settings">
              <label htmlFor="brush-slider">
                <i className="fa-solid fa-paint-brush"></i> Brush Size
              </label>
              <div className="slider-wrapper">
                <input
                  type="range"
                  id="brush-slider"
                  min="8"
                  max="24"
                  value={brushSize}
                  onChange={(e) => setBrushSize(parseInt(e.target.value))}
                />
                <span id="brush-value">{brushSize}px</span>
              </div>
            </div>

            <div className="button-group">
              <button id="clear-btn" className="btn btn-secondary" onClick={clearCanvas}>
                <i className="fa-solid fa-eraser"></i> Clear
              </button>
              <button
                id="scan-btn"
                className="btn btn-primary"
                onClick={handlePrediction}
                disabled={isScanning}
              >
                <i className="fa-solid fa-microscope"></i> Scan Digit
              </button>
            </div>
          </div>

          <div className="settings-panel">
            <label className="toggle-switch">
              <input
                type="checkbox"
                id="auto-predict-toggle"
                checked={autoPredict}
                onChange={(e) => setAutoPredict(e.target.checked)}
              />
              <span className="slider"></span>
              <span className="label-text">Auto-predict on release</span>
            </label>
          </div>

          {error && (
            <div className="error-banner">
              <i className="fa-solid fa-triangle-exclamation"></i>
              <span>{error}</span>
            </div>
          )}
        </section>

        {/* Analysis & Results Panel */}
        <section className="card results-card" aria-labelledby="results-heading">
          <div className="card-header">
            <h2 id="results-heading">
              <i className="fa-solid fa-chart-simple"></i> Classification Engine
            </h2>
            <p className="subtitle">Real-time neural network predictions & activation map.</p>
          </div>

          <div className="results-grid">
            {/* Top Prediction View */}
            <div className="prediction-view">
              <div className="indicator-title">Detected Digit</div>
              <div
                id="digit-display"
                className={`digit-display ${prediction !== null ? 'detected' : 'idle'} ${
                  isScanning ? 'scanning' : ''
                }`}
              >
                <span id="prediction-digit">{prediction !== null ? prediction : '-'}</span>
              </div>
              <div className="confidence-container">
                <span className="label">Confidence:</span>
                <span id="confidence-percentage" className="confidence-val">
                  {confidence !== null ? `${(confidence * 100).toFixed(2)}%` : '--%'}
                </span>
              </div>
            </div>

            {/* Preprocessed input display */}
            <div className="preview-view">
              <div className="indicator-title">Network Input (28×28)</div>
              <div className="preview-box">
                <img
                  id="preprocess-preview"
                  src={
                    preprocessedImage ||
                    "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='28' height='28'><rect width='28' height='28' fill='%2311121d'/></svg>"
                  }
                  alt="Model Input Preview"
                />
                <div className="grid-overlay"></div>
              </div>
              <div className="preview-caption">Inverted & normalized</div>
            </div>
          </div>

          {/* Softmax probabilities chart */}
          <div className="probabilities-section">
            <h3>
              <i className="fa-solid fa-list-ol"></i> Class Probabilities (Softmax Distribution)
            </h3>
            <div id="probabilities-chart" className="probabilities-chart">
              {probabilities.map((prob, index) => {
                const isWinning = prediction === index;
                return (
                  <div key={index} className="probability-row">
                    <span className={`digit-num ${isWinning ? 'winning' : ''}`}>{index}</span>
                    <div className="bar-container">
                      <div
                        className={`bar-fill ${isWinning ? 'winning' : ''}`}
                        style={{ width: `${(prob * 100).toFixed(1)}%` }}
                      ></div>
                    </div>
                    <span className={`prob-val ${isWinning ? 'winning' : ''}`}>
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </section>
      </main>

      <footer className="main-footer">
        <p>Neural Digit Scanner Dashboard • Powered by TensorFlow & Keras</p>
      </footer>
    </div>
  );
}

export default App;
